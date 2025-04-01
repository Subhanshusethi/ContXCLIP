from IPython import display as ipythondisplay
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import BertModel, BertConfig, BertTokenizer
import albumentations as A
import cv2
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import time
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np  
import json
from torch.utils.data import DataLoader , Dataset
from PIL import Image
import json
import numpy as np  
from collections import Counter
import spacy
import pickle
from transformers import CLIPModel, CLIPProcessor,get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import pickle


print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

for i in range(5):
    tokens, mask, prefix,caption = dataset[i]
    # print(dataset.prefix_length)
    print(f"Maximum sequence length: {dataset.max_seq_len}")
    print(f"Element {i + 1}:")
    print(f"Tokens: {tokens.shape}")
    print(f"Mask: {mask}")
    print(f"Caption:{caption}")
    # prefix = prefix.unsqueeze(0)
    print(f"Prefix Embedding: {prefix.size(0)}")
    print("-" * 80)


class ImageEncoder(nn.Module):
    """
    Encodes image and returns it's embedding.
    """

    def __init__(self, model='openai/clip-vit-base-patch32', device="cuda"):
        super(ImageEncoder, self).__init__()

        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)
        # self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model(**image)

        return image_features.pooler_output

class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model =CFG.text_encoder_model , device=CFG.device):
        super(TextDecoder, self).__init__()

        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.target_token_idx =0

        self.model =AutoModelForCausalLM.from_pretrained(model).to(self.device)
        # self.model.eval()
        self.vocab_size = self.model.config.vocab_size

        self.freeze_layers()
        
        
    def freeze_layers(self):
        for p in [
            *list(self.model.parameters()),
        ]:  # Freeze everything except the last two layers
            p.requires_grad = False
    

    def forward(self, embedding=None, attention_mask=None,input_ids = None,labels=None,train_mode=False):
        assert (embedding is None) != (input_ids is None), "Provide either embedding or input_ids, not both."
        text_features = self.model(
            inputs_embeds=embedding, input_ids=input_ids,labels=labels, attention_mask=attention_mask)
        logits = text_features.logits
        return logits

class ClipModel(nn.Module):
    def __init__(self,ep_len,projection_dim):
        super().__init__()
        self.ie = ImageEncoder()
        self.td = TextDecoder()
        self.ip_tg = ImageProjection(ep_len=ep_len,projection_dim=projection_dim)
        self.tp_tg = TextProjection(projection_dim=projection_dim)
        self.ip_ig = ImageProjection(ep_len=ep_len,projection_dim=projection_dim)
        self.tp_ig = TextProjection(projection_dim=projection_dim)
        self.criteria = nn.CrossEntropyLoss(ignore_index=0)
        self.MLP_FINAL = MLP(embedding_dim=projection_dim*2, hidden_dims=[CFG.text_embedding], dropout=0.4)
        self.MLP_INTER = MLP(embedding_dim=projection_dim, hidden_dims=[128,projection_dim], dropout=0.4)
        self.ep_len = ep_len
        self.max_len = CFG.max_length
        self.temperature = 1
        # self.fc_out = nn.Linear(CFG.image_embedding, CFG.text_embedding)
        # self.norm = nn.LayerNorm( CFG.text_embedding)
        self.final_emb_norm = nn.LayerNorm(512)
        # self.norm_2 = nn.LayerNorm( 512)
        self.lstm_norm = nn.LayerNorm(512)
        self.enhance_layer_tg = nn.LayerNorm(projection_dim)  # Example layer
        self.enhance_layer_ig = nn.LayerNorm(projection_dim)  # Example layer
        self.lstm_1 = nn.LSTM(input_size=projection_dim,hidden_size=projection_dim,num_layers=2,batch_first=True,dropout=0.4,bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=projection_dim,hidden_size=projection_dim,num_layers=2,batch_first=True,dropout=0.3,bidirectional=True)
        d_p = 512  # Or choose an appropriate value less than projection_dim
        self.xgl_attention = XGLAttentionSeq(embedding_dim=512, d_p=d_p)
        self.xgl_attention_txt = XGLAttentionSeq(embedding_dim=256, d_p=256)
        self.xgl_att_img = XGLAttentionSeq(embedding_dim = 256,d_p = 256)
        # self.out_norm= nn.LayerNorm(50257)
        #example use of lstm_1
        # lstm_1_out, _ = self.lstm_1(final_emb)

        # self.td.model.transformer.wte.weight = self.td.model.lm_head.weight
#         self.enhance_layer = EFFN(projection_dim=projection_dim)
    def calculate_logits(self, embedding1, embedding2):
        embedding2_transposed = embedding2.transpose(1, 2)  # Shape: (batch_size, projection_dim, seq_length2)
        logits = torch.matmul(embedding1, embedding2_transposed) / self.temperature  # Shape: (batch_size, seq_length1, seq_length2)
        return logits
    
    def enhance_embed_txt(self, embedding1, embedding2, logits):
        attn_weights = torch.softmax(logits, dim=-1)  # Softmax over seq_length2
        output = torch.matmul(attn_weights, embedding2)  # Shape: (batch_size, seq_length1, projection_dim)
        enhanced_embedding = embedding1 + output
        enhanced_embedding = self.enhance_layer_tg(enhanced_embedding)
        return enhanced_embedding
    def enhance_embed_img(self, embedding1, embedding2, logits):
        attn_weights = torch.softmax(logits, dim=-1)  # Softmax over seq_length2
        output = torch.matmul(attn_weights, embedding2)  # Shape: (batch_size, seq_length1, projection_dim)
        enhanced_embedding = embedding1 + output
        enhanced_embedding = self.enhance_layer_ig(enhanced_embedding)
        return enhanced_embedding
    def top_k_logits(self,logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    def forward(self, img_emb, trg_cap, att_mask):
        # print(f"Initial img_emb shape: {img_emb.shape}")
        # print(f"Initial trg_cap shape: {trg_cap.shape}")
        # print(f"Initial att_mask shape: {att_mask.shape}")
    
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]  # (batch_size, trg_len-1)
        y = trg_cap[:, 1:]  # Target captions shifted
        # print(f"x shape: {x.shape}, x_mask shape: {x_mask.shape}")
        # print(f"y shape: {y.shape}")

    # Image projection for text-guided (TG)
        image_projection_tg = self.ip_tg(img_emb, train=True)  # (batch_size, ep_len, projection_dim)
        # print(f"image_projection_tg shape: {image_projection_tg.shape}")
        image_projection_tg = self.xgl_att_img(image_projection_tg)  # Apply attention
        # print(f"image_projection_tg after XGLAttention shape: {image_projection_tg.shape}")

    # Text projection for TG
        text_projection = self.td.model.transformer.wte(x)  # Embedding layer
        # print(f"text_projection (from wte) shape: {text_projection.shape}")
        text_projection_tg = self.tp_tg(text_projection)  # Projected text embeddings
        # print(f"text_projection_tg shape: {text_projection_tg.shape}")
        text_projection_tg = self.xgl_attention_txt(text_projection_tg)
        # print(f"text_projection_tg after XGLAttention shape: {text_projection_tg.shape}")

    # Calculate TG logits
        logits_tg = self.calculate_logits(image_projection_tg, text_projection_tg)  # (batch_size, seq_length1, seq_length2)
        # print(f"logits_tg shape: {logits_tg.shape}")
        enhanced_embedding_tg = self.enhance_embed_txt(image_projection_tg, text_projection_tg, logits_tg)  # Enhanced embeddings
        # print(f"enhanced_embedding_tg shape: {enhanced_embedding_tg.shape}")

    # Image projection for image-guided (IG)
        image_projection_ig = self.ip_ig(img_emb, train=True)  # (batch_size, ep_len, projection_dim)
        # print(f"image_projection_ig shape: {image_projection_ig.shape}")
        image_projection_ig = self.xgl_att_img(image_projection_ig)
        # print(f"image_projection_ig after XGLAttention shape: {image_projection_ig.shape}")

    # Text projection for IG
        text_projection_ig = self.tp_ig(text_projection)  # (batch_size, trg_len-1, projection_dim)
        # print(f"text_projection_ig shape: {text_projection_ig.shape}")
        text_projection_ig = self.xgl_attention_txt(text_projection_ig)
        # print(f"text_projection_ig after XGLAttention shape: {text_projection_ig.shape}")

    # Calculate IG logits
        logits_ig = self.calculate_logits(text_projection_ig, image_projection_ig)
        # print(f"logits_ig shape: {logits_ig.shape}")
        enhanced_embedding_ig = self.enhance_embed_img(text_projection_ig, image_projection_ig, logits_ig)
        # print(f"enhanced_embedding_ig shape: {enhanced_embedding_ig.shape}")
        # padded_tensor = torch.nn.functional.pad(enhanced_embedding_tg, (0, 0, 20, 0), mode='constant', value=0)
        # logits_new = torch.matmul(enhanced_embedding_ig, padded_tensor.transpose(1, 2))  # Shape: (batch, seq_len1, seq_len2)

        # image_similarity = self.calculate_logits(enhanced_embedding_ig, enhanced_embedding_ig)
        # text_similarity = self.calculate_logits(padded_tensor,padded_tensor)
        # targets = F.softmax(
            # (image_similarity + text_similarity) / 2 * self.temperature, dim=-1
        # )
        # texts_loss = self.criteria (logits_new, targets)
        # images_loss = self.criteria (logits_new.T, targets.T)
        # loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # print(loss)
        # print(f"enhanced_embedding_ig shape: {enhanced_embedding_ig.shape}")

    # LSTMs
        lstm_layer_1, _ = self.lstm_1(self.MLP_INTER(torch.cat([enhanced_embedding_ig, enhanced_embedding_tg], dim=1)))
        # print(f"lstm_layer_1 shape: {lstm_layer_1.shape}")
        lstm_layer_2, _ = self.lstm_2(self.MLP_INTER(torch.cat([enhanced_embedding_tg, enhanced_embedding_ig], dim=1)))
        # print(f"lstm_layer_2 shape: {lstm_layer_2.shape}")

    # Combine LSTM outputs
        final_emb = lstm_layer_1 + lstm_layer_2
        # print(f"final_emb shape after LSTMs: {final_emb.shape}")
        final_emb = self.lstm_norm(final_emb)  # Normalize
        # print(f"final_emb after normalization shape: {final_emb.shape}")

    # Attention on final embedding
        final_emb_att = self.xgl_attention(final_emb)
        # print(f"final_emb_att shape: {final_emb_att.shape}")
        final_emb = final_emb + final_emb_att
        # final_emb = self.final_emb_norm(final_emb)
        # print(f"final_emb after final norm shape: {final_emb.shape}")

    # Final output projection
        final_emb = self.MLP_FINAL(final_emb)
        # print(f"final_emb after fc_out shape: {final_emb.shape}")

    # Attention mask update
        x_mask = x_mask.long()
        # print(f"x_mask shape after concatenation: {x_mask.shape}")

    # Positional embeddings
        pos_emb = self.td.model.transformer.wpe(
            torch.arange(final_emb.shape[1]).to(CFG.device)
        )
        pos_emb = pos_emb.expand_as(final_emb)
        final_emb = final_emb + pos_emb
        # print(f"final_emb after adding positional embeddings shape: {final_emb.shape}")

    # Decoder output
        output = self.td(embedding=final_emb, attention_mask=x_mask)
        # print(f"Decoder output shape: {output.shape}")

    # Truncate output for loss calculation
        output = output[:, self.ep_len:, : -1]
        # print(f"output after truncation shape: {output.shape}")

    # Calculate loss
        loss = self.criteria(output.reshape(-1, output.shape[-1]), y.flatten())
        # print(f"Loss: {loss.item()}")
        return loss

    # def evaluate(self, img_emb=None, max_len=None, image=None, temperature=1, beam_width=5):
    #     # self.eval()
    #     if img_emb is None and image is not None:
    #         img_emb = self.ie(image)
    #     elif img_emb is None:
    #         raise ValueError("Either img_emb or image must be provided.")

    #     if max_len is None:
    #         max_len = self.max_len
    #     if temperature is None:
    #         temperature = self.temperature

    #     with torch.no_grad():
    #     # Image projections
    #         image_projection_tg = self.ip_tg(img_emb, train=True)  # (batch_size, ep_len, projection_dim)
    #         image_projection_ig = self.ip_ig(img_emb, train=True)  # (batch_size, ep_len, projection_dim)

    #         batch_size = img_emb.size(0)

    #     # Initialize beams
    #         generated = torch.full(
    #             (batch_size, 1, 1),
    #             self.td.tokenizer.bos_token_id,
    #             dtype=torch.long,
    #             device=CFG.device,
    #         )  # (batch_size, beam_width, seq_len)
    #         scores = torch.zeros(batch_size, 1, device=CFG.device)  # (batch_size, beam_width)

    #     # Expand image projections to match beam width
    #         image_projection_tg = image_projection_tg.unsqueeze(1).repeat(1, beam_width, 1, 1)
    #         image_projection_ig = image_projection_ig.unsqueeze(1).repeat(1, beam_width, 1, 1)

    #         for _ in range(max_len):
    #             batch_size, beam_width_current, seq_len = generated.size()

    #         # Flatten beams
    #             generated_flat = generated.view(batch_size * beam_width_current, seq_len)
    #             image_projection_tg_flat = image_projection_tg.view(
    #                 batch_size * beam_width_current, -1, image_projection_tg.size(-1)
    #             )
    #             image_projection_ig_flat = image_projection_ig.view(
    #                 batch_size * beam_width_current, -1, image_projection_ig.size(-1)
    #             )

    #         # Text embeddings
    #             text_embeddings = self.td.model.transformer.wte(generated_flat)

    #         # Text projections
    #             text_projection_tg = self.tp_tg(text_embeddings)
    #             text_projection_ig = self.tp_ig(text_embeddings)

    #         # Calculate logits and enhanced embeddings
    #             logits_tg = self.calculate_logits(text_projection_tg, image_projection_tg_flat)
    #             enhanced_embedding_tg = self.enhance_embed(text_projection_tg, image_projection_tg_flat, logits_tg)

    #             logits_ig = self.calculate_logits(image_projection_ig_flat, text_projection_ig)
    #             enhanced_embedding_ig = self.enhance_embed(image_projection_ig_flat, text_projection_ig, logits_ig)

    #         # Combine embeddings
    #             final_emb = torch.cat([enhanced_embedding_ig, enhanced_embedding_tg], dim=1)
    #             final_emb = self.fc_out(final_emb)

    #         # Positional embeddings
    #             pos_emb = self.td.model.transformer.wpe(
    #                 torch.arange(final_emb.shape[1]).to(CFG.device)
    #             )
    #             pos_emb = pos_emb.unsqueeze(0).expand_as(final_emb)
    #             final_emb = final_emb + pos_emb

    #         # Attention mask
    #             attention_mask = torch.ones(final_emb.size(0), final_emb.size(1)).to(CFG.device)

    #         # Pass through the decoder
    #             output = self.td(embedding=final_emb, attention_mask=attention_mask)
    #             next_token_logits = output[:, -1, :]  # (batch_size * beam_width_current, vocab_size)
    #             next_token_log_probs = torch.log_softmax(next_token_logits / temperature, dim=-1)

    #         # Update scores
    #             scores = scores.view(batch_size * beam_width_current, 1)
    #             total_scores = scores + next_token_log_probs  # (batch_size * beam_width_current, vocab_size)

    #         # Reshape for topk
    #             total_scores = total_scores.view(batch_size, -1)  # (batch_size, beam_width_current * vocab_size)
    #             top_scores, top_indices = total_scores.topk(beam_width, dim=-1)  # (batch_size, beam_width)

    #         # Compute beam and token indices
    #             beam_indices = top_indices // next_token_log_probs.size(-1)
    #             token_indices = top_indices % next_token_log_probs.size(-1)

    #         # Gather previous sequences
    #             generated = generated.view(batch_size, beam_width_current, -1)
    #             batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(beam_indices)
    #             prev_generated = generated[batch_indices, beam_indices, :]

    #         # Append new tokens
    #             next_tokens = token_indices.unsqueeze(-1)
    #             generated = torch.cat([prev_generated, next_tokens], dim=-1)

    #         # Update scores
    #             scores = top_scores

    #         # Check for EOS tokens (optional)
    #             eos_mask = (next_tokens == self.td.tokenizer.eos_token_id)
    #             if eos_mask.all():
    #                 break

    #     # Select the best sequences
    #         best_scores, best_indices = scores.max(dim=-1)
    #         best_sequences = generated[torch.arange(batch_size), best_indices, 1:]  # Exclude BOS token

    #         return best_sequences

    def evaluate(self,img_emb=None,max_len=None,image=None,temperature=1,second=False):
        self.eval()
        if second:
            img_emb = self.ie(image)
            print(f"img_emb shape:    {img_emb.shape}")
        else:
            img_emb = img_emb

        if max_len is None:
            max_len = self.max_len
        if temperature is None:
            temperature = self.temperature

        with torch.no_grad():
            # Image projection
            image_projection_tg = self.ip_tg(img_emb,train=True)  # (batch_size, ep_len, projection_dim)
            image_projection_tg = self.xgl_att_img(image_projection_tg)
            image_projection_ig = self.ip_ig(img_emb,train=True)  # (batch_size, ep_len, projection_dim)
            image_projection_ig = self.xgl_att_img(image_projection_ig)

            batch_size = img_emb.size(0)

            # Initialize generated tokens with BOS token
            generated = torch.full(
                (batch_size, 1),
                self.td.tokenizer.bos_token_id,
                dtype=torch.long,
                device=CFG.device,
            )

            # Initialize attention mask
            attention_mask = torch.ones(batch_size, 1).to(CFG.device)
            for _ in range(max_len):
#                 print(generated)
#                 print("hereeeeeeeee")


                # Text embeddings for generated tokens
                text_embeddings = self.td.model.transformer.wte(generated) 
                # text_embeddings = self.xgl_attention_txt(text_embeddings)# (batch_size, seq_len, d_model)
                text_projection_tg = self.tp_tg(text_embeddings)  # (batch_size, seq_len, projection_dim)
                text_projection_tg  = self.xgl_attention_txt(text_projection_tg)
                
#                 print("here1")
                # Self-Attention
                # image_projection_red_t = self.text_guided_sai(image_projection_tg)  # (batch_size, projection_dim)
                # text_projection_red_t = self.text_guided_sat(text_projection_tg)  # (batch_size, projection_dim)
#                 print("here2")
                # Calculate logits and enhance embeddings
                logits_tg = self.calculate_logits(text_projection_tg,image_projection_tg)
                enhanced_embedding_tg = self.enhance_embed_img(
                   text_projection_tg,image_projection_tg,  logits_tg
                )  # (batch_size, projection_dim)
#                 print("here3")
                # Cross-Attention and FFN for text
                # tg_attention = self.textguidedattention(enhanced_embedding_tg, text_projection_red_t)  # (batch_size, projection_dim)
                # tg_attention = tg_attention.unsqueeze(1)  # (batch_size, 1, projection_dim)
                # tg_attention = torch.cat([text_projection, tg_attention], dim=1)  # (batch_size, seq_len + 1, projection_dim)
                tg_ffn = enhanced_embedding_tg  # (batch_size, seq_len + 1, projection_dim)
#                 print("here4")
                # Self-Attention on image projection
                text_projection_ig = self.tp_ig(text_embeddings)  # (batch_size, seq_len, projection_dim)
                # image_projection_red_i = self.image_guided_sai(image_projection_ig)  # (batch_size, projection_dim)
                # text_projection_red_i = self.image_guided_sat(text_projection_ig)  # (batch_size, projection_dim)
                logits_ig = self.calculate_logits(image_projection_ig , text_projection_ig )
                enhanced_embedding_ig = self.enhance_embed_txt(
                    image_projection_ig , text_projection_ig , logits_ig
                )  # (batch_size, projection_dim)

                # Cross-Attention and FFN for image
                # ig_attention = self.imageguidedattention(enhanced_embedding_ig, image_projection_red_i)  # (batch_size, projection_dim)
                # ig_attention = ig_attention.unsqueeze(1)  # (batch_size, 1, projection_dim)
                # ig_attention = torch.cat([image_projection, ig_attention], dim=1)  # (batch_size, ep_len + 1, projection_dim)
                ig_ffn = enhanced_embedding_ig  # (batch_size, ep_len + 1, projection_dim)

                # Concatenate image and text features
                lstm_layer_1,_ = self.lstm_1(self.MLP_INTER(torch.cat([ig_ffn, tg_ffn], dim=1))) #(batch_size, trg_len-1, projection_dim)
                lstm_layer_2,_ = self.lstm_2(self.MLP_INTER(torch.cat([tg_ffn, ig_ffn], dim=1))) #(batch_size, ep_len+1, projection_dim)
                final_emb = lstm_layer_1 + lstm_layer_2 # (batch_size, ep_len + seq_len + 1, projection_dim)
                final_emb = self.xgl_attention(final_emb)
                final_emb = self.MLP_FINAL(final_emb)
                # final_emb = self.norm(final_emb)# (batch_size, ep_len + seq_len + 1, d_model)
                

                # Positional embeddings
                pos_emb = self.td.model.transformer.wpe(
                    torch.arange(final_emb.shape[1]).to(CFG.device)
                )  # (ep_len + seq_len + 1, d_model)
                pos_emb = pos_emb.unsqueeze(0).expand_as(final_emb)  # (batch_size, ep_len + seq_len + 1, d_model)
                final_emb = final_emb + pos_emb  # (batch_size, ep_len + seq_len + 1, d_model)

                # Update attention mask
                x_mask = torch.cat(
                    [torch.ones(batch_size, ig_ffn.shape[1]).to(CFG.device), attention_mask], dim=1
                )  # (batch_size, total_seq_len)

                # Pass through the text decoder
                output = self.td(embedding=final_emb, attention_mask=x_mask)
#                 print("outputshape:  ",output.shape)# (batch_size, total_seq_len, vocab_size)

                # Get the logits for the next token
                next_token_logits = output[:, -1, :]
        
#                 print(next_token_logits)# (batch_size, vocab_size)
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)

                # Greedy decoding (you can use sampling or beam search here)
                next_token = torch.argmax(probs, dim=-1).unsqueeze(1)  # (batch_size, 1)

                # Append the generated token
                generated = torch.cat([generated, next_token], dim=1)  # (batch_size, seq_len + 1)
#                 print(generated)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1).to(CFG.device)], dim=1
                )  # (batch_size, seq_len + 1)

                # Stop if all sequences have generated the EOS token
                if torch.all(next_token == self.td.tokenizer.eos_token_id):
                    break

            # Return generated tokens (excluding the initial BOS token)
            return generated[:, 1:]
    

    def evaluate(self, img_emb=None, max_len=None, image=None, temperature=1, top_k=50):
    # self.eval()
        if img_emb is None and image is not None:
            img_emb = self.ie(image)
        elif img_emb is None:
            raise ValueError("Either img_emb or image must be provided.")

        if max_len is None:
            max_len = self.max_len
        attn_store = {
        "text_proj_tg_wt": [],
        "text_proj_xgl_tg_wt": [],
        "text_proj_ig_wt": [],
        "text_proj_xgl_ig_wt": [],
        "image_proj_tg_wt": [],
        "image_proj_xgl_tg_wt": [],
        "image_proj_ig_wt": [],
        "image_proj_xgl_ig_wt": [],
        "final_emb_wt": []
        }

        with torch.no_grad():
            image_projection_tg,image_proj_tg_wt = self.ip_tg(img_emb,train=True,return_wt=True)  # (batch_size, ep_len, projection_dim)
            print(f"shape of image_proj_tg_wt is {image_proj_tg_wt.shape}")
            attn_store["image_proj_tg_wt"].append(image_proj_tg_wt.detach().cpu())  # Store or append
            image_projection_tg,image_proj_xgl_tg_wt = self.xgl_att_img(image_projection_tg,return_wt=True)
            print(f"shape of image_proj_xgl_tg_wt is {image_proj_xgl_tg_wt.shape}")
            attn_store["image_proj_xgl_tg_wt"].append(image_proj_xgl_tg_wt.detach().cpu())
            image_projection_ig,image_proj_ig_wt = self.ip_ig(img_emb,train=True,return_wt=True)  # (batch_size, ep_len, projection_dim)
            print(f"shape of image_proj_ig_wt is {image_proj_ig_wt.shape}")

            attn_store["image_proj_ig_wt"].append(image_proj_ig_wt.detach().cpu())  # NEW

            image_projection_ig,image_proj_xgl_ig_wt = self.xgl_att_img(image_projection_ig,return_wt=True)
            print(f"shape of image_proj_xgl_ig_wt is {image_proj_xgl_ig_wt.shape}")
            attn_store["image_proj_xgl_ig_wt"].append(image_proj_xgl_ig_wt.detach().cpu())  # NEW


            batch_size = img_emb.size(0)
            generated = torch.full(
                (batch_size, 1),
                self.td.tokenizer.bos_token_id,
                dtype=torch.long,
                device=CFG.device,
            )

            attention_mask = torch.ones(batch_size, 1).to(CFG.device)
            generated_tokens = []

            for _ in range(max_len):
                text_embeddings = self.td.model.transformer.wte(generated)
                # text_embeddings = self.xgl_attention_txt(text_embeddings)
                text_projection_tg,text_proj_tg_wt = self.tp_tg(text_embeddings,return_wt=True)
                
                attn_store["text_proj_tg_wt"].append(text_proj_tg_wt.detach().cpu())  # NEW

                text_projection_tg,text_proj_xgl_tg_wt  = self.xgl_attention_txt(text_projection_tg,return_wt = True)
                attn_store["text_proj_xgl_tg_wt"].append(text_proj_xgl_tg_wt.detach().cpu())  # NEW

                logits_tg = self.calculate_logits(text_projection_tg, image_projection_tg)
                enhanced_embedding_tg = self.enhance_embed_img(
                    text_projection_tg, image_projection_tg, logits_tg
                )
                tg_ffn = enhanced_embedding_tg

                text_projection_ig, text_proj_ig_wt = self.tp_ig(text_embeddings,return_wt=True)
                attn_store["text_proj_ig_wt"].append(text_proj_ig_wt.detach().cpu())  # NEW
                text_projection_ig, text_proj_xgl_ig_wt = self.xgl_attention_txt(text_projection_ig,return_wt=True)
                attn_store["text_proj_xgl_ig_wt"].append(text_proj_xgl_ig_wt.detach().cpu())  # NEW
                logits_ig = self.calculate_logits(image_projection_ig, text_projection_ig)
                enhanced_embedding_ig = self.enhance_embed_txt(
                    image_projection_ig, text_projection_ig, logits_ig
                )
                ig_ffn = enhanced_embedding_ig

                lstm_layer_1, _ = self.lstm_1(self.MLP_INTER(torch.cat([ig_ffn, tg_ffn], dim=1)))
                lstm_layer_2, _ = self.lstm_2(self.MLP_INTER(torch.cat([tg_ffn, ig_ffn], dim=1)))
                final_emb = lstm_layer_1 + lstm_layer_2
                final_emb, final_emb_wt= self.xgl_attention(final_emb,return_wt = True)
                attn_store["final_emb_wt"].append(final_emb_wt.detach().cpu())  # NEW
                final_emb = self.MLP_FINAL(final_emb)

                seq_length = final_emb.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=CFG.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
                pos_emb = self.td.model.transformer.wpe(position_ids)
                final_emb = final_emb + pos_emb

                x_mask = torch.cat(
                    [torch.ones(batch_size, ig_ffn.shape[1]).to(CFG.device), attention_mask], dim=1
                )

                output = self.td(embedding=final_emb, attention_mask=x_mask)
                next_token_logits = output[:, -1, :] / temperature

            # Apply top-k sampling
                filtered_logits = self.top_k_logits(next_token_logits, top_k)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1).to(CFG.device)], dim=1
                )

                generated_tokens.append(next_token.item())

                if torch.all(next_token == self.td.tokenizer.eos_token_id):
                    break

            return generated[:, 1:],attn_store





from torchinfo import summary
# Instantiate your model
model = ClipModel(
    ep_len=4,
    projection_dim=CFG.projection_dim
)

# Define a dummy input that matches the expected input of your model
# For example, suppose your image embeddings have shape (batch_size, image_embedding_dim)
# and your target captions are of shape (batch_size, seq_len)

# Replace these with the actual dimensions
batch_size = 1 # or any batch size you want to test with
image_embedding_dim = CFG.image_embedding
seq_len = CFG.max_length
num_tokens = seq_len  # assuming the sequence length matches the number of tokens
vocab_size = 50257 # same as num_emb used in your model

# Create dummy inputs
dummy_img_emb = torch.randn(batch_size, image_embedding_dim).to(CFG.device)
dummy_trg_cap = torch.randint(0, vocab_size, (batch_size, seq_len)).to(CFG.device)
dummy_att_mask = torch.ones(batch_size, seq_len).to(CFG.device)

# Since your model's `forward` method requires three inputs, we'll create an input data tuple
input_data = (dummy_img_emb, dummy_trg_cap, dummy_att_mask)
model.to_empty(device=CFG.device)
# Now, use torchinfo.summary
summary(
    model,
    input_data=input_data,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=8,
    device=CFG.device  # Add this line
)



# del model
# gc.collect()
# torch.cuda.empty_cache()


def build_loaders(dataset,tokenizer,mode):
    # tokenizer = tokenizer
    return DataLoader(dataset,
                      batch_size=10,
                      # collate_fn= lambda b: cl_fn(b,tokenizer),
                      shuffle=True if mode == "train" else False,
                      num_workers=CFG.num_workers,
                      # pin_memory=True,
                      drop_last=True,
    )

if __name__ == "__main__":
    main()

import torch
import cv2
import os
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu

def evaluate(dataset, model, tokenizer, image_path=None, save_path=None):
    model.eval()
    loop = tqdm(dataset, total=len(dataset))
    
    # Lists to store BLEU scores for averaging later
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for idx, (tokens, mask, prefix, caption) in enumerate(loop):
        with torch.no_grad():
            img_embed = prefix.float().to(CFG.device)
            img_embed = img_embed.unsqueeze(0)

            predicted_tokens = model.evaluate(img_emb=img_embed, temperature=1)
            predicted_captions = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            predicted_caption = predicted_captions[0] if len(predicted_captions) > 0 else ""

            # Tokenize reference and predicted
            reference_tokens = caption.strip().split()
            predicted_tokens = predicted_caption.strip().split()
            references = [reference_tokens]

            # Calculate BLEU scores for n=1 to 4
            bleu1 = sentence_bleu(references, predicted_tokens, weights=(1.0, 0, 0, 0))
            bleu2 = sentence_bleu(references, predicted_tokens, weights=(0.5, 0.5, 0, 0))
            bleu3 = sentence_bleu(references, predicted_tokens, weights=(1/3, 1/3, 1/3, 0))
            bleu4 = sentence_bleu(references, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))

            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)

            print(f"Predicted: {predicted_caption}")
            print(f"Reference: {caption}")
            print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")

    # Compute average BLEU scores over the entire dataset
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0

    print("----- AVERAGE BLEU SCORES -----")
    print(f"Average BLEU-1: {avg_bleu1:.4f}")
    print(f"Average BLEU-2: {avg_bleu2:.4f}")
    print(f"Average BLEU-3: {avg_bleu3:.4f}")
    print(f"Average BLEU-4: {avg_bleu4:.4f}")


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.pad_token = "0"
    
    validdata = test_data
    model = ClipModel(ep_len=10, projection_dim=CFG.projection_dim).to(device='cuda')
    model_path = "/kaggle/working/best.pt"
    checkpoint = torch.load(model_path, map_location=CFG.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    dataset = validdata
    img_path = "/kaggle/input/coco-image-caption/train2014/train2014"
    save_path = '/kaggle/working/main_folder_13'
    evaluate(dataset, model, tokenizer, img_path, save_path)


def overlay_attention_on_image(image_bgr, attn_map, alpha=0.5):
    """
    Overlays a 2D attention map on top of an original BGR image using OpenCV.

    Args:
        image_bgr (np.ndarray): Original image in BGR format with shape (H, W, 3).
        attn_map (np.ndarray): Attention map with shape (h, w). Will be resized to (H, W).
        alpha (float): Blending factor. 0 = only original image, 1 = only heatmap.
    
    Returns:
        overlayed (np.ndarray): The resulting image with heatmap overlay in BGR format.
    """
    # 1) Resize attention map to the same size as the image
    H, W, _ = image_bgr.shape
    attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2) Normalize attention map to [0, 1]
    attn_norm = attn_resized - attn_resized.min()
    denom = (attn_resized.max() - attn_resized.min()) + 1e-8
    attn_norm = attn_norm / denom  # shape (H, W) in [0, 1]

    # 3) Convert to 8-bit and apply a color map
    attn_255 = (attn_norm * 255).astype(np.uint8)      # (H, W)
    heatmap  = cv2.applyColorMap(attn_255, cv2.COLORMAP_JET)  # (H, W, 3) in BGR

    # 4) Alpha blend heatmap with original image
    overlayed = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

    return overlayed

def evaluate(dataset,model,tokenizer,image_path=None,save_path=None):
#     model.load_state_dict(torch.load("model"),map_location = CFG.device)
    model.eval()
    loop = tqdm(dataset , total = len(test_data))
    
    for idx, (tokens, mask, prefix,caption,img_name) in enumerate(loop):
#         print(caption)

#         try:
        with torch.no_grad():
            img_path_full = os.path.join(image_path, img_name)
            image = cv2.imread(img_path_full)
            
#             image_emb = torch.tensor(image).to(CFG.device)
#             image_emb = image_emb.unsqueeze(0)
#                 print("hello")
            # img_embed = torch.tensor(img_emb).float().to(CFG.device)
            img_embed = prefix.float().to(CFG.device)
            # print(img_embed.shape)
            img_embed = img_embed.unsqueeze(0)
            tokens, attn_store = model.evaluate(img_emb=img_embed, temperature=1)
            image_proj_tg_wt     = attn_store["image_proj_tg_wt"]       # shape [1, 10, 10]
            image_proj_xgl_tg_wt = attn_store["image_proj_xgl_tg_wt"]   # shape [1, 10]
            image_proj_ig_wt     = attn_store["image_proj_ig_wt"]       # shape [1, 10, 10]
            image_proj_xgl_ig_wt = attn_store["image_proj_xgl_ig_wt"]   # shape [1, 10]

# 1) Visualize the 2D matrix (10x10)
            matrix_tg = image_proj_tg_wt[0].cpu().numpy()
            matrix_tg = matrix_tg.squeeze(0)
            # plt.figure()
            # plt.imshow(matrix_tg, cmap='Blues', aspect='auto')
            # plt.colorbar()
            # plt.title("image_proj_tg_wt - 10x10")
            # plt.show()

# 2) Visualize the 1D vector (10)
            vector_tg_xgl = image_proj_xgl_tg_wt[0].cpu().numpy()
            vector_tg_xgl = vector_tg_xgl.squeeze(0)
            # plt.figure()
            # plt.bar(range(len(vector_tg_xgl)), vector_tg_xgl)
            # plt.title("image_proj_xgl_tg_wt - 10")
            # plt.show()
# 
#             tokens = model.evaluate(image=image_emb,second=True)
#             print(tokens)
            captions = tokenizer.batch_decode(
            tokens, skip_special_tokens=True)
            print("captions:      ",captions)
#                 cap_list.append(decoded + "\n")
            # plt.imshow(image)
            # plt.title(captions)
            # plt.axis("off")
            # plt.savefig(os.path.join(save_path, img_name), bbox_inches="tight")
            # plt.clf()
            # plt.close()
#         except IndexError :
#             pass   
#         except KeyboardInterrupt:
#             print("Intreupptrd by user")
#             break
        
        
if __name__ == "__main__":
    tokenizer =  tokenizer = GPT2Tokenizer.from_pretrained("/kaggle/input/gpt-finetuned-final-ver")
    tokenizer.bos_token = tokenizer.eos_token
    # tokenizer.pad_token = "0"
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    validdata = test_data
    model =ClipModel(ep_len=10, projection_dim=CFG.projection_dim).to(device ='cuda')
#     model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Use GPU 0 and 1
#     model = model.to('cuda')
    model_path = "/kaggle/working/best.pt"
#     model.load_state_dict(torch.load(model_path, map_location=CFG.device, weights_only=True))
    checkpoint = torch.load(model_path, map_location=CFG.device)
    
    # Load the model state dictionary
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    dataset = validdata
    img_path = "/kaggle/input/coco-image-caption/train2014/train2014"
    save_path = '/kaggle/working/main_folder_13'
    evaluate(dataset,model,tokenizer,img_path,save_path)
                
    

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate(dataset, model, tokenizer, image_path=None, save_path=None):
    model.eval()
    loop = tqdm(dataset, total=len(dataset))

    # Create subfolders for TG and IG weights
    os.makedirs(os.path.join(save_path, "tg"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "ig"), exist_ok=True)

    for idx, (tokens, mask, prefix, caption) in enumerate(loop):

        with torch.no_grad():
            # Move image prefix embedding to GPU if needed
            img_embed = prefix.float().to(CFG.device)
            img_embed = img_embed.unsqueeze(0)

            # Evaluate the model; returns generated tokens + attn_store
            tokens, attn_store = model.evaluate(img_emb=img_embed, temperature=1)

            # -----------
            # GET WEIGHTS
            # -----------
            # Each shape = [1, 10, 10] for batch_size=1
            image_proj_tg_wt = attn_store["image_proj_tg_wt"]
            image_proj_ig_wt = attn_store["image_proj_ig_wt"]

            # -----------------------------------------------------------------
            # VISUALIZE TG PROJECTION WEIGHTS (10x10) AND SAVE TO tg/ FOLDER
            # -----------------------------------------------------------------
            matrix_tg = image_proj_tg_wt[0].cpu().numpy()  # shape (1, 10, 10)
            matrix_tg = matrix_tg.squeeze(0)               # shape (10, 10)

            plt.figure()
            plt.imshow(matrix_tg, cmap='Blues', aspect='auto')
            plt.colorbar()
            plt.title(f"image_proj_tg_wt - 10x10 (sample {idx})")

            save_file_tg = os.path.join(save_path, "tg", f"{idx}_image_proj_tg_wt.png")
            plt.savefig(save_file_tg, bbox_inches="tight")
            plt.close()

            # -----------------------------------------------------------------
            # VISUALIZE IG PROJECTION WEIGHTS (10x10) AND SAVE TO ig/ FOLDER
            # -----------------------------------------------------------------
            matrix_ig = image_proj_ig_wt[0].cpu().numpy()  # shape (1, 10, 10)
            matrix_ig = matrix_ig.squeeze(0)               # shape (10, 10)

            plt.figure()
            plt.imshow(matrix_ig, cmap='Reds', aspect='auto')
            plt.colorbar()
            plt.title(f"image_proj_ig_wt - 10x10 (sample {idx})")

            save_file_ig = os.path.join(save_path, "ig", f"{idx}_image_proj_ig_wt.png")
            plt.savefig(save_file_ig, bbox_inches="tight")
            plt.close()

            # -----------------------------------------------------------------
            # DECODE TOKENS -> CAPTIONS
            # -----------------------------------------------------------------
            captions = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            print("captions:", captions)

    # End of evaluate function

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("/kaggle/input/gpt-finetuned-final-ver")
    tokenizer.bos_token = tokenizer.eos_token
    validdata = test_data

    model = ClipModel(ep_len=10, projection_dim=CFG.projection_dim).to('cuda')
    checkpoint = torch.load("/kaggle/working/best.pt", map_location=CFG.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on your dataset
    img_path = "/kaggle/input/coco-image-caption/train2014/train2014"
    save_path = "/kaggle/working/main_folder_13"
    evaluate(validdata, model, tokenizer, img_path, save_path)


def overlay_attention_on_image(image_bgr, attn_map, alpha=0.5):
    """
    Overlays a 2D attention map on top of an original BGR image using OpenCV.

    Args:
        image_bgr (np.ndarray): Original image in BGR format with shape (H, W, 3).
        attn_map (np.ndarray): Attention map with shape (h, w). Will be resized to (H, W).
        alpha (float): Blending factor. 0 = only original image, 1 = only heatmap.
    
    Returns:
        overlayed (np.ndarray): The resulting image with heatmap overlay in BGR format.
    """
    # 1) Resize attention map to the same size as the image
    H, W, _ = image_bgr.shape
    attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2) Normalize attention map to [0, 1]
    attn_norm = attn_resized - attn_resized.min()
    denom = (attn_resized.max() - attn_resized.min()) + 1e-8
    attn_norm = attn_norm / denom  # shape (H, W) in [0, 1]

    # 3) Convert to 8-bit and apply a color map
    attn_255 = (attn_norm * 255).astype(np.uint8)      # (H, W)
    heatmap  = cv2.applyColorMap(attn_255, cv2.COLORMAP_JET)  # (H, W, 3) in BGR

    # 4) Alpha blend heatmap with original image
    overlayed = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

    return overlayed

def evaluate(dataset,model,tokenizer,image_path=None,save_path=None):
#     model.load_state_dict(torch.load("model"),map_location = CFG.device)
    model.eval()
    loop = tqdm(dataset , total = len(test_data))
    
    for idx, (tokens, mask, prefix,caption,img_name) in enumerate(loop):
#         print(caption)

#         try:
        with torch.no_grad():
            img_id_str = f"{int(img_name):012d}"
            # 1) Load the image with OpenCV
            img_path_full =os.path.join(image_path, f"COCO_train2014_{img_id_str}.jpg")
            image_bgr = cv2.imread(img_path_full)  # shape: (H, W, 3) in BGR
            if image_bgr is None:
                print(f"Could not read image: {img_path_full}")
                continue
            
#             image_emb = torch.tensor(image).to(CFG.device)
#             image_emb = image_emb.unsqueeze(0)
#                 print("hello")
            # img_embed = torch.tensor(img_emb).float().to(CFG.device)
            img_embed = prefix.float().to(CFG.device)
            # print(img_embed.shape)
            img_embed = img_embed.unsqueeze(0)
            tokens, attn_store = model.evaluate(img_emb=img_embed, temperature=5,sampling_method="top_n_sigma")
            image_proj_tg_wt     = attn_store["image_proj_tg_wt"]       # shape [1, 10, 10]
            image_proj_xgl_tg_wt = attn_store["image_proj_xgl_tg_wt"]   # shape [1, 10]
            image_proj_ig_wt     = attn_store["image_proj_ig_wt"]       # shape [1, 10, 10]
            image_proj_xgl_ig_wt = attn_store["image_proj_xgl_ig_wt"]   # shape [1, 10]

# 1) Visualize the 2D matrix (10x10)
            matrix_tg = image_proj_tg_wt[0].cpu().numpy()
            matrix_tg = matrix_tg.squeeze(0)
            overlayed_image = overlay_attention_on_image(image_bgr, matrix_tg, alpha=0.5)
            overlayed_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
            # plt.figure()
            # plt.imshow(matrix_tg, cmap='Blues', aspect='auto')
            # plt.colorbar()
            # plt.title("image_proj_tg_wt - 10x10")
            # plt.show()

# 2) Visualize the 1D vector (10)
            vector_tg_xgl = image_proj_xgl_tg_wt[0].cpu().numpy()
            vector_tg_xgl = vector_tg_xgl.squeeze(0)
            # plt.figure()
            # plt.bar(range(len(vector_tg_xgl)), vector_tg_xgl)
            # plt.title("image_proj_xgl_tg_wt - 10")
            # plt.show()
# 
#             tokens = model.evaluate(image=image_emb,second=True)
#             print(tokens)
            captions = tokenizer.batch_decode(
            tokens, skip_special_tokens=True)
            print("captions:      ",captions)
            if save_path is not None:
                print("heyy")
                # e.g., "overlayed_<idx>.jpg" or incorporate the original filename
                save_fname = f"{os.path.splitext(img_name)[0]}_overlayed.jpg"
                save_full_path = os.path.join(save_path, save_fname)
                # cv2.imwrite(save_full_path, overlayed_image)  # writes in BGR
#                 cap_list.append(decoded + "\n")
            plt.imshow( overlayed_image)
            plt.title(captions)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, img_name), bbox_inches="tight")
            plt.clf()
            plt.close()
#         except IndexError :
#             pass   
#         except KeyboardInterrupt:
#             print("Intreupptrd by user")
#             break
        
        
if __name__ == "__main__":
    tokenizer =  tokenizer = GPT2Tokenizer.from_pretrained("/kaggle/input/gpt-finetuned-final-ver")
    tokenizer.bos_token = tokenizer.eos_token
    # tokenizer.pad_token = "0"
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    validdata = test_data
    model =ClipModel(ep_len=10, projection_dim=CFG.projection_dim).to(device ='cuda')
#     model = torch.nn.DataParallel(model, device_ids=[0, 1])  # Use GPU 0 and 1
#     model = model.to('cuda')
    model_path = "/kaggle/working/best.pt"
#     model.load_state_dict(torch.load(model_path, map_location=CFG.device, weights_only=True))
    checkpoint = torch.load(model_path, map_location=CFG.device)
    
    # Load the model state dictionary
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    dataset = validdata
    img_path = "/kaggle/input/coco-image-caption/train2014/train2014"
    save_path = '/kaggle/working/folder3'
    evaluate(dataset,model,tokenizer,img_path,save_path)
                
    

os.makedirs('/kaggle/working/folder3')


