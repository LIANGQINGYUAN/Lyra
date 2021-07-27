# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

from transformer import TransformerLayer, PositionalEmbedding, LayerNorm
from word_prob_layer import WordProbLayer
from label_smoothing import LabelSmoothing

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()  

        self.config = config
        self.device = config.device
        self.copy = config.pointer_gen
        self.d_model = config.d_model #embeding size
        self.enc_dict_size = config.enc_vocab_size #vocab_size
        self.dec_dict_size = config.dec_vocab_size #vocab_size
        self.enc_pad_token_idx = 0 #pad id
        self.dec_pad_token_idx = 1 #pad id
        self.num_layers = config.n_layers #num layers
        self.d_ff = config.d_inner #d_inner
        self.num_heads = config.n_head #num heads
        self.dropout = config.dropout #dropout
        self.smoothing_factor = config.smoothing_factor #label_smoothing

        self.enc_tok_embed = nn.Embedding(self.enc_dict_size, self.d_model, self.enc_pad_token_idx)
        self.dec_tok_embed = nn.Embedding(self.dec_dict_size, self.d_model, self.dec_pad_token_idx)
        self.enc_pos_embed = PositionalEmbedding(self.d_model)
        self.dec_pos_embed = PositionalEmbedding(self.d_model)

        self.enc_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.enc_layers.append(TransformerLayer(self.d_model, self.d_ff, self.num_heads, self.dropout))
        
        self.dec_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dec_layers.append(TransformerLayer(self.d_model, self.d_ff, self.num_heads, self.dropout, with_external=True))

        self.emb_layer_norm = LayerNorm(self.d_model)

        self.word_prob = WordProbLayer(self.d_model, self.dec_dict_size, self.device, self.copy, self.dropout)

        self.smoothing = LabelSmoothing(self.device, self.dec_dict_size, self.dec_pad_token_idx, self.smoothing_factor)
       
        self.init_weights()

    def init_weights(self):
        init_uniform_weight(self.enc_tok_embed.weight)
        init_uniform_weight(self.dec_tok_embed.weight)

    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx)
        #return torch.cat([torch.gt(seq[:, 1],0).unsqueeze(1), torch.lt(seq[:, 1:], 0)], 1)

    def get_subsequent_mask(self, seq):
        ''' For masking out the subsequent info. '''
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1))#.bool()
        return subsequent_mask

    def cal_loss(self, y_pred, y):
        loss = - torch.log(torch.gather(y_pred.clamp(min=1e-8), -1, y.unsqueeze(-1))).squeeze(-1).masked_fill_((y == self.dec_pad_token_idx), 0.).mean(-1).mean()
        return loss

    def label_smotthing_loss(self, y_pred, y, y_padding_mask):
        bsz, seq_len = y.size()
        y_pred = T.log(y_pred.clamp(min=1e-8))
        loss = self.smoothing(y_pred.view(seq_len * bsz, -1), y.contiguous().view(seq_len * bsz, -1))
        return loss / T.sum(y_padding_mask)

    def encode(self, inp, mask_x):
        x = self.enc_tok_embed(inp) + self.enc_pos_embed(inp)
        for layer_id, layer in enumerate(self.enc_layers):
            x = layer(x,x,x, mask=mask_x)
        return x

    def decode(self, y_inp, mask_y, enc_output, mask_x, x_ext=None, max_ext_len=None):
        dec_embedding = self.dec_tok_embed(y_inp) + self.dec_pos_embed(y_inp)
        dec_output = dec_embedding
        for layer_id, layer in enumerate(self.dec_layers):
            dec_output = layer(query=dec_output, key=enc_output, value=enc_output, mask=mask_y, src_mask=mask_x)
        
        if self.copy:
            #dec_output, dec_embedding, enc_output
            y_pred = self.word_prob(dec_output, dec_embedding, enc_output, mask_x, x_ext, max_ext_len)
        else:
            y_pred = self.word_prob(dec_output)
       
        return y_pred

    def forward(self, x, y_inp, y_tgt, x_ext, max_ext_len):

        mask_x = self.get_pad_mask(x, self.enc_pad_token_idx)
        mask_y = self.get_subsequent_mask(y_inp)

        # print("***************** Encoder start *****************")
        enc_output = self.encode(x, mask_x)
        # print("***************** Encoder end *****************")

        # print("***************** Decoder start *****************")
        if self.copy:
            y_pred = self.decode(y_inp, mask_y, enc_output, mask_x, x_ext, max_ext_len)
        else:
            y_pred = self.decode(y_inp, mask_y, enc_output, mask_x)
        # cost = self.cal_loss(y_pred, y_tgt)
        cost = self.label_smotthing_loss(y_pred, y_tgt, self.get_pad_mask(y_tgt, self.dec_pad_token_idx))
        # print("***************** Decoder end *****************")
        
        return y_pred, cost
    
