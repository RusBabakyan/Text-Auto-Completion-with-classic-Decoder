import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.key = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.value = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, S, D = x.shape
        query = self.query(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        key =   self.key(x).  view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        dots = (query @ key.transpose(-1, -2)) * self.scale


        mask = torch.tril(torch.ones((S, S))).to(x.device)
        dots.masked_fill_(mask == 0, float('-inf'))

        att_scores = dots.softmax(-1)
        att_v = att_scores @ value

        out = att_v.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        out = self.out(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.MHA = MultiHeadAttention(d_model, n_heads, qkv_bias)
        self.MLP = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):

        x = self.layer_norm_1(self.MHA(x)) + x
        x = self.layer_norm_2(self.MLP(x)) + x

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_blocks, qkv_bias = True, seq_len = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, seq_len)
        self.blocks = nn.Sequential(*[DecoderBlock(d_model, n_heads, qkv_bias) for _ in range(n_blocks)])
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.to(next(iter(self.parameters())).device)
        emb = self.embedding(x)
        emb = self.pos(emb)

        x = self.blocks(emb)

        out = self.final_layer(x)

        return out