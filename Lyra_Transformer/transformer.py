import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TransformerLayer(nn.Module):
    
    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = LayerNorm(embed_dim)
        self.ff_layer_norm = LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout)
            self.external_layer_norm = LayerNorm(embed_dim)

    def forward(self, query, key, value, mask=None, src_mask=None):
        # query: bsz x seq_len x embed_dim
        residual = query

        #encoder and decoder self attention
        x = self.self_attn(query, query, query, mask)
        #Add and Norm for attention
        x = F.dropout(x, p=self.dropout)
        x = self.attn_layer_norm(residual + x)
        
        #encoder-decoder attention
        if self.with_external:
            residual = x
            x = self.external_attn(x, key, value, src_mask)
            x = F.dropout(x, p=self.dropout)
            x = self.attn_layer_norm(residual + x)
        
        #feed forward
        residual = x
        x = self.fc1(x)
        x = F.dropout(x, p=self.dropout)
        x = gelu(self.fc2(x))
        x = F.dropout(x, p=self.dropout)
        x = self.ff_layer_norm(residual + x)
        

        return x


class MultiheadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, need_attn=False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if need_attn:
            return self.output_linear(x), attn
        else:
            return self.output_linear(x)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            #src mask
            if len(list(mask.size())) == 2:
                mask = mask.unsqueeze(1).repeat(1, query.size(2), 1).unsqueeze(1) 
            #trg mask
            elif len(list(mask.size())) == 3:
                mask = mask.unsqueeze(1)
            #print(mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            attn_value = dropout(torch.matmul(p_attn, value))
        else:
            attn_value = torch.matmul(p_attn, value)
        
        return attn_value, p_attn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-1)]
