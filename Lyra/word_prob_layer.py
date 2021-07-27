# -*- coding: utf-8 -*-
#pylint: skip-file
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from transformer import MultiheadAttention

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, dict_size, device, copy, dropout):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.dropout = dropout
    
        if self.copy:
            self.external_attn = MultiheadAttention(self.hidden_size, 1, self.dropout)
            self.proj = nn.Linear(self.hidden_size * 3, self.dict_size)
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size * 3))
            self.bv = nn.Parameter(torch.Tensor(1))
        else:
            self.proj = nn.Linear(self.hidden_size, self.dict_size)

    def forward(self, dec_output, y_emb=None, memory=None, mask_x=None, xids=None, max_ext_len=None):
        if self.copy:
            atts, dists = self.external_attn(query=dec_output, key=memory, value=memory, mask=mask_x, need_attn=True)
            pred = T.softmax(self.proj(T.cat([dec_output, y_emb, atts], -1)), dim=-1)
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(pred.size(0), pred.size(1), max_ext_len)).to(self.device)
                pred = T.cat((pred, ext_zeros), -1)
            g = T.sigmoid(F.linear(T.cat([dec_output, y_emb, atts], -1), self.v, self.bv))
            xids = xids.unsqueeze(0).repeat(pred.size(1), 1, 1) .transpose(0, 1)
           
            pred = (g * pred).scatter_add(2, xids, (1 - g) * dists.squeeze())
        else:
            pred = T.softmax(self.proj(dec_output), dim=-1)
        return pred
