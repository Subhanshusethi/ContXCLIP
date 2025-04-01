import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from config import CFG
from model import ReluSIG

class ImageProjection(nn.Module):
    def __init__(self, ep_len, projection_dim, embedding_dim=CFG.image_embedding):
        super(ImageProjection, self).__init__()
        self.ep_len = ep_len
        self.projection_dim = projection_dim
        
        # First linear projection from embedding_dim to ep_len * projection_dim
        self.projection_1 = nn.Linear(embedding_dim, self.ep_len * self.projection_dim)
        
        # Second linear layer added here
        # self.projection_2 = nn.Linear(self.projection_dim, CFG.text_embedding)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.gelu = nn.GELU()
        self.gelusig = ReluSIG()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        self.mhai = nn.MultiheadAttention(
            embed_dim=self.projection_dim, num_heads=8, batch_first=True, dropout=0.2
        )

    def forward(self, x, train,return_wt = False):
        # First linear projection
        x = self.projection_1(x)
        x = self.gelusig(x)
        x = self.dropout(x)
        
        # Reshape based on training or evaluation mode
        x = (
            x.view(-1, self.ep_len, self.projection_dim)
            if train
            else x.view(self.ep_len, self.projection_dim)
        )
        x_out,x_wt = self.mhai(x,x,x)
            
        # x_out = self.transformer_encoder(x)
        
        x_out = self.layer_norm(x+x_out)
        if return_wt:
            return x_out,x_wt
        else:
            return x_out


def train_epoch(model, train_loader, optimizer, lr_scheduler=None):
    model.train()
    plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    loss_meter = AvgMeter()
    scaler = torch.cuda.amp.GradScaler()  # GradScaler for mixed precision
    tqdm_object = tqdm(train_loader, total=len(train_loader), mininterval=5)
    history = []

    for idx, (tokens, mask, prefix, _,_) in enumerate(tqdm_object):
        # Move tensors to device
        tokens, mask, prefix = tokens.to(CFG.device), mask.to(CFG.device), prefix.to(CFG.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward and backward pass with mixed precision
        with torch.autocast("cuda"):
            loss = model(prefix, tokens, mask)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)
        scaler.step(optimizer)
        scaler.update()

        # Step LR scheduler if applicable
        if lr_scheduler:
            lr_scheduler.step()

        # Update loss meter and history
        count = prefix.size(0)
        loss_meter.update(loss.item(), count)
        history.append(loss.item())

        # Update progress bar
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    # Plot training history
    plotter.plot(history)

    return loss_meter


def main():
    import os

    train_df = train_data 
    valid_df = test_data

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CFG.text_encoder_model, add_bos_token=True)
    tokenizer.bos_token = tokenizer.eos_token
    # tokenizer.pad_token = "0"

    # Build data loaders
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="Valid")

    # Initialize the model
    model = ClipModel(ep_len=10, projection_dim=CFG.projection_dim).to(CFG.device)

    # Define the optimizer and learning rate scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # Only parameters with requires_grad=True will be updated
   
    params = [
    # ImageProjection layers
        {"params": itertools.chain(model.ip_tg.parameters(), model.ip_ig.parameters()), "lr": 5e-5},
    
    # TextProjection layers
        {"params": itertools.chain(model.tp_tg.parameters(), model.tp_ig.parameters()), "lr": 5e-5},
    
    # XGLAttention layers
        # {"params": model.td.parameters(), "lr": 5e-5},
        {"params": model.xgl_attention.parameters(), "lr": 1e-4},
        {"params": model.xgl_attention_txt.parameters(), "lr": 1e-4},
        {"params": model.xgl_att_img.parameters(), "lr": 1e-4},
    
    # LSTM layers
        {"params": model.lstm_1.parameters(), "lr": 1e-4},
        {"params": model.lstm_2.parameters(), "lr": 1e-4},
    
    # MLP layers
        {"params": model.MLP_FINAL.parameters(), "lr": 1e-4},
        {"params": model.MLP_INTER.parameters(), "lr": 1e-4},
    
    # Final Linear layer
        # {"params": model.fc_out.parameters(), "lr": 5e-5}
    ]
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)


    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5000, num_training_steps=CFG.epochs * len(train_loader)
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    # )

    # Check if a checkpoint exists
    if os.path.exists('checkpoin5t.pt'):
        checkpoint = torch.load('checkpoijnt.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        train_history = checkpoint['train_history']
        valid_history = checkpoint['valid_history']
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
    else:
        start_epoch = 0
        best_loss = float('inf')
        train_history = []
        valid_history = []

    try:
        for epoch in range(start_epoch, CFG.epochs):
            print(f"Epoch: {epoch + 1}")

            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler=lr_scheduler)
            train_history.append(train_loss.avg)

            # Save checkpoint after every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_loss': best_loss,
                    'train_history': train_history,
                    'valid_history': valid_history,
                }, 'checkpoint.pt')
                print(f"Checkpoint saved at epoch {epoch + 1}")

            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader)
                valid_history.append(valid_loss.avg)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                # Save the best model separately
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, 'best.pt')
                print("Saved Best Model!")
                # lr_scheduler.step(valid_loss.avg)

    except KeyboardInterrupt:
        print("Interrupted by user")
        # Optionally save a checkpoint upon interruption
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'train_history': train_history,
            'valid_history': valid_history,
        }, 'checkpoint.pt')
        print(f"Checkpoint saved at epoch {epoch + 1} due to interruption")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Training loss')
    plt.plot(valid_history, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



