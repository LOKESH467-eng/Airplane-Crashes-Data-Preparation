"""
Aviation SLM V3
Original model architecture extracted from the original V3 training notebook.

Architecture:
- Decoder-only Transformer
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention
- SwiGLU feed-forward network
- Tied input and output embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
CONFIG = {'vocab_size': 151669, 'd_model': 256, 'n_layers': 6, 'n_heads': 4, 'n_kv_heads': 2, 'd_ff': 768, 'max_seq_len': 512, 'dropout': 0.1, 'rope_theta': 10000.0, 'gradient_checkpointing': True, 'tie_word_embeddings': True}

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)

def apply_rope(q, k, theta=10000.0):
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError('RoPE requires an even head dimension.')
    device = q.device
    frequencies = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    cos = angles.cos()[None, None, :, :].to(q.dtype)
    sin = angles.sin()[None, None, :, :].to(q.dtype)

    def rotate(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
    return (rotate(q), rotate(k))

class GroupedQueryAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        d = config['d_model']
        heads = config['n_heads']
        kv_heads = config['n_kv_heads']
        self.n_heads = heads
        self.n_kv_heads = kv_heads
        self.head_dim = d // heads
        self.dropout = config['dropout']
        self.rope_theta = config['rope_theta']
        self.q_proj = nn.Linear(d, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        batch, seq_len, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_theta)
        repeat_count = self.n_heads // self.n_kv_heads
        if repeat_count > 1:
            k = k.repeat_interleave(repeat_count, dim=1)
            v = v.repeat_interleave(repeat_count, dim=1)
        attention_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch, seq_len, d)
        return self.o_proj(attention_output)

class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        d = config['d_model']
        ff = config['d_ff']
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        d = config['d_model']
        self.attention_norm = RMSNorm(d)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(d)
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x

class AviationSLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.token_embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layers'])])
        self.final_norm = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        if config['tie_word_embeddings']:
            self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._initialize_weights)
        if config['tie_word_embeddings']:
            self.lm_head.weight = self.token_embeddings.weight

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.token_embeddings(input_ids)
        x = self.embedding_dropout(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embeddings.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config['vocab_size']), targets.reshape(-1))
        return (logits, loss)
