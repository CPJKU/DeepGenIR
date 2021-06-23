'''
code is taken from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking
'''

from typing import Dict, Iterator, List
import pickle
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.nn import util

import math


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # expects (S,N,E)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TK(Model):

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_intermediate_size: int):

        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = torch.tensor(kernels_mu, dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = torch.tensor(kernels_sigma, dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        _embsize = self.word_embeddings.get_output_dim()
        
        self._pos_embedding = PositionalEncoding(_embsize, dropout=0)

        encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_intermediate_size, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)
 
        #self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
        #         hidden_dim=_embsize,
        #         projection_dim=att_proj_dim,
        #         feedforward_hidden_dim=att_intermediate_size,
        #         num_layers=att_layer,
        #         num_attention_heads=att_heads,
        #         dropout_prob = 0,
        #         residual_dropout_prob = 0,
        #         attention_dropout_prob = 0)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_embeddings = self.word_embeddings(query)
        document_embeddings = self.word_embeddings(document)

        # we assume 1 is the unknown token, 0 is padding - both need to be removed

        if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
            query_pad_oov_mask = (query["tokens"] > 1).float()
            document_pad_oov_mask = (document["tokens"] > 1).float()
        else: # == 3 (elmo characters per word)
            query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
            document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
        
        query_embeddings_pos = self._pos_embedding(query_embeddings.transpose(0,1))
        query_embeddings_pos = query_embeddings_pos.transpose(0,1) * query_pad_oov_mask.unsqueeze(-1)
        query_embeddings_context = self.forward_representation(query_embeddings_pos, query_pad_oov_mask)
        
        enc_start_time = time.time()
        document_embeddings_pos = self._pos_embedding(document_embeddings.transpose(0,1))
        document_embeddings_pos = document_embeddings_pos.transpose(0,1) * document_pad_oov_mask.unsqueeze(-1)
        document_embeddings_context = self.forward_representation(document_embeddings_pos, document_pad_oov_mask)
        enc_time = time.time() - enc_start_time

        
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings_context, document_embeddings_context)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        return {"rels":score, "enc_time":enc_time}

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        seq_embeddings_context = self.contextualizer(sequence_embeddings.transpose(0,1), 
                                                     src_key_padding_mask=~sequence_mask.bool()).transpose(0,1)
        seq_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * seq_embeddings_context) * sequence_mask.unsqueeze(-1)
        
        return seq_embeddings

    def cuda(self, device=None):
        self = super().cuda(device)
        self.mu = self.mu.cuda(device)
        self.sigma = self.sigma.cuda(device)
        return self 


