import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from tqdm import tqdm

from config import CFG
from model import ReluSIG


class TextProjection(nn.Module):
    def __init__(self, projection_dim, embedding_dim=CFG.text_embedding):
        super(TextProjection, self).__init__()
        self.projection_t = nn.Linear(embedding_dim, projection_dim)
        self.mhat = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.gelu = nn.GELU()
        self.gelusig = ReluSIG()
        self.dropout = nn.Dropout(0.35)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x, return_wt=False):
        x = self.projection_t(x)
        x = self.gelusig(x)
        x_out, x_wt = self.mhat(x, x, x)
        x_out = self.dropout(x_out)
        x_out = x + x_out
        x_out = self.layer_norm(x_out)
        return (x_out, x_wt) if return_wt else x_out


class EFFN(nn.Module):
    def __init__(self, projection_dim):
        super(EFFN, self).__init__()
        self.ffn = nn.Linear(projection_dim, projection_dim)
        self.gelu = nn.GELU()
        self.gelusig = ReluSIG()
        self.dropout = nn.Dropout(0.01)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        x_out = self.ffn(x)
        x_out = self.gelusig(x_out)
        x_out = self.dropout(x_out)
        x_out = x + x_out
        x_out = self.layer_norm(x_out)
        return x_out


class CrossAttention(nn.Module):
    def __init__(self, projection_dim):
        super(CrossAttention, self).__init__()
        self.mhca = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        x_out, _ = self.mhca(x, y, y)
        x_out = x + x_out
        x_out = self.layer_norm(x_out)
        return x_out


class XGLAttentionSeq(nn.Module):
    def __init__(self, embedding_dim, d_p):
        super(XGLAttentionSeq, self).__init__()
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.gelusig = ReluSIG()
        self.omega_k_q = nn.Linear(embedding_dim, d_p)
        self.omega_k = nn.Linear(embedding_dim, d_p)
        self.omega_v = nn.Linear(embedding_dim, d_p)
        self.omega_p = nn.Linear(d_p, d_p)
        self.omega_b = nn.Linear(d_p, 1)
        self.batch_norm = nn.BatchNorm1d(d_p)
        self.sse_fc1 = nn.Linear(d_p, d_p // 2)
        self.sse_fc2 = nn.Linear(d_p // 2, d_p)
        self.sigmoid = nn.Sigmoid()
        self.fc_final = nn.Linear(d_p, d_p)

    def forward(self, x, return_wt=False):
        batch_size, seq_len, embedding_dim = x.size()
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q_proj = F.gelu(self.omega_k_q(Q))
        K_proj = F.gelu(self.omega_k(K))
        P_k = Q_proj * K_proj

        S = F.relu(self.omega_p(P_k))
        S = self.omega_b(S).squeeze(-1)
        S_normalized = F.softmax(S, dim=-1)

        V_proj = self.gelusig(self.omega_v(V))
        P_v_k = Q_proj * V_proj
        F_weighted = S_normalized.unsqueeze(-1) * P_v_k

        B = self.batch_norm(P_k.view(-1, P_k.size(-1))).view(batch_size, seq_len, -1)
        B_sse = self.sigmoid(self.sse_fc2(F.relu(self.sse_fc1(B))))
        F_final = F_weighted * B_sse

        F_output = self.fc_final(F_final)
        return (F_output, S_normalized) if return_wt else F_output


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        input_dim = embedding_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim)
            ])
            input_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale
        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()
            if self.scale is None:
                plt.plot(data)
            elif self.scale == 'semilogx':
                plt.semilogx(data)
            elif self.scale == 'semilogy':
                plt.semilogy(data)
            elif self.scale == 'loglog':
                plt.loglog(data)
            else:
                raise ValueError("Unrecognized scale: {}".format(self.scale))
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())
            self.tic = time.time()


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def valid_epoch(model, valid_loader):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), mininterval=5)

    with torch.no_grad():
        for idx, (tokens, mask, prefix, *_rest) in enumerate(tqdm_object):
            tokens, mask, prefix = tokens.to(CFG.device), mask.to(CFG.device), prefix.to(CFG.device)
            with torch.autocast("cuda"):
                loss = model(prefix, tokens, mask)
            count = prefix.size(0)
            loss_meter.update(loss.item(), count)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter
