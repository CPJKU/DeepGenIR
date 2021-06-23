'''
code is taken from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking
'''

from typing import Dict, Iterator, List
import pickle
import numpy as np
from collections import defaultdict
import math

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.nn import util

from transformers import BertModel


class BertKernel(Model):

    def __init__(self,
                 bert: BertModel,
                 kernels_mu: List[float],
                 kernels_sigma: List[float]):

        super().__init__(None)

        self._bert = bert
        self._embedding_size = self._bert.config.hidden_size 


        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = torch.tensor(kernels_mu, dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = torch.tensor(kernels_sigma, dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.cosine_module = CosineMatrixAttention()

        self.dense = nn.Linear(self._embedding_size + n_kernels, 1, bias=False)
        #self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        #self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        #
        # prepare BERT embedding
        # -------------------------------------------------------

        bsz = document["tokens"].size(0)
        doc_max_length = document["tokens"].size(1)
        qry_max_length = query["tokens"].size(1)
        
        tok_seq = torch.full((bsz, qry_max_length + doc_max_length + 3), 0, dtype=int).cuda()
        seg_mask = torch.full((bsz, qry_max_length + doc_max_length + 3), 0, dtype=int).cuda()
        seg_1_value = 0
        seg_2_value = 1
        CLS_id = 101
        SEP_id = 102
        
        tok_seq[:, 0] = torch.full((bsz, 1), CLS_id, dtype=int)[:, 0]
        seg_mask[:, 0] = torch.full((bsz, 1), seg_1_value, dtype=int)[:, 0]
        
        for batch_i in range(bsz):
            # query
            _offset = 1
            _vec = query["tokens"][batch_i]
            _length = len(_vec[_vec != 0])
            tok_seq[batch_i, _offset:_length+_offset] = query["tokens"][batch_i, :_length]
            seg_mask[batch_i, _offset:_length+_offset] = torch.full((_length, 1), seg_1_value, dtype=int)[:, 0]
            _offset += _length
            
            tok_seq[batch_i, _offset:_offset+1] = SEP_id
            seg_mask[batch_i, _offset:_offset+1] = seg_1_value
            _offset += 1
            
            # document
            _vec = document["tokens"][batch_i]
            _length = len(_vec[_vec != 0])
            tok_seq[batch_i, _offset:_length+_offset] = document["tokens"][batch_i, :_length]
            seg_mask[batch_i, _offset:_length+_offset] = torch.full((_length, 1), seg_2_value, dtype=int)[:, 0]
            _offset += _length
            
            tok_seq[batch_i, _offset:_offset+1] = SEP_id
            seg_mask[batch_i, _offset:_offset+1] = seg_2_value
            _offset += 1
            
        pad_mask = util.get_text_field_mask({"tokens":tok_seq}).cuda()

        bert_out = self._bert(input_ids=tok_seq, attention_mask=pad_mask, token_type_ids=seg_mask)
        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        _query_mask = util.get_text_field_mask(query) # 1 - not masked, 0 - masked;
        _query_segs = _query_mask.new_zeros(_query_mask.size()).detach().requires_grad_(False)
        query_embeddings_context = self._bert(input_ids=query["tokens"], token_type_ids=_query_segs, attention_mask=_query_mask, return_dict=True)['last_hidden_state']
        
        _document_mask = util.get_text_field_mask(document) # 1 - not masked, 0 - masked;
        _document_segs = _document_mask.new_zeros(_document_mask.size()).detach().requires_grad_(False)
        document_embeddings_context = self._bert(input_ids=document["tokens"], token_type_ids=_document_segs, attention_mask=_document_mask, return_dict=True)['last_hidden_state']
        
        # we assume 1 is the unknown token, 0 is padding - both need to be removed

        if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
            query_pad_oov_mask = (query["tokens"] > 1).float()
            document_pad_oov_mask = (document["tokens"] > 1).float()
        else: # == 3 (elmo characters per word)
            query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
            document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
        
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

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #
        # mean kernels
        #
        
        #per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        #log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        #log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        #per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        features = torch.cat([bert_out[0][:, 0, :], per_kernel], dim=1)
        dense_out = self.dense(features)
        #dense_mean_out = self.dense_mean(per_kernel_mean)
        #dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.tanh(torch.squeeze(dense_out, 1)) #torch.tanh(dense_out), 1)

        return {"rels":score}

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

