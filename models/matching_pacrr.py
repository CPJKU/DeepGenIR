import pickle, sys
from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import numpy as np
from scipy.special import softmax

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn                              
import torch.nn.functional as F

import utils

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention                          


class PACRR(Model):
    '''
    Paper: PACRR: A Position-Aware Neural IR Model for Relevance Matching, Hui et al., EMNLP'17

    Reference code (but in tensorflow):
    
    * first-hand: https://github.com/khui/copacrr/blob/master/models/pacrr.py
    
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 unified_query_length:int,
                 unified_document_length:int,
                 max_conv_kernel_size: int, # 2 to n
                 conv_output_size: int, # conv output channels
                 kmax_pooling_size: int, # per query k-max pooling
                 idfcf_path: str) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings
        self.cosine_module = CosineMatrixAttention()
        #self.cosine_module = DotProductMatrixAttention()

        self.unified_query_length = unified_query_length
        self.unified_document_length = unified_document_length


        self.convolutions = []
        for i in range(2, max_conv_kernel_size + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=i, in_channels=1, out_channels=conv_output_size), # this outputs [batch,32,unified_query_length,unified_document_length]
                    nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        self.kmax_pooling_size = kmax_pooling_size

        self.dense = nn.Linear((kmax_pooling_size * max_conv_kernel_size + 1) * unified_query_length ,
                               out_features=100, bias=True)
        self.dense2 = nn.Linear(100, out_features=10, bias=True)
        self.dense3 = nn.Linear(10, out_features=1, bias=False)
        
        ## loading IDF values
        self.idf_lookup = self.get_idf_lookup(idfcf_path, vocab)        

    def get_idf_lookup(self, idfcf_path, vocab):
        ## loading IDF values
        with open (idfcf_path, 'rb') as fr:
            idfcf_dic = pickle.load(fr)

        _idfs_unnorm = []
        for v_i in range(vocab.get_vocab_size()):
            v = vocab.get_token_from_index(v_i)
            if v not in idfcf_dic:
                continue
            _idfs_unnorm.append(idfcf_dic[v][0])

        _idfs_unnorm = np.array(_idfs_unnorm).reshape(-1, 1)
        _scaler = MinMaxScaler()
        _scaler.fit(_idfs_unnorm)
        _idfs = _scaler.transform(_idfs_unnorm).squeeze(-1)

        _idfs_vocab = []
        _idfs_vocab_cnt = 0
        for v_i in range(vocab.get_vocab_size()):
            v = vocab.get_token_from_index(v_i)
            if v in idfcf_dic:
                _idfs_vocab.append(_idfs[_idfs_vocab_cnt])
                _idfs_vocab_cnt += 1
            else:
                if v_i in [0, 1]: #padding and unknown
                    _idfs_vocab.append(0.0)
                else: # other words
                    _idfs_vocab.append(0.01) # some low value
        idf_lookup = torch.nn.Embedding(vocab.get_vocab_size(), 1)
        idf_lookup.weight = nn.Parameter(torch.Tensor(_idfs_vocab), requires_grad=False)

        return idf_lookup
    
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
            # shape: (batch, query_max)
            query_pad_oov_mask = (query["tokens"] > 1).float()
            # shape: (batch, doc_max)
            document_pad_oov_mask = (document["tokens"] > 1).float()
        else: # == 3 (elmo characters per word)
            # shape: (batch, query_max)
            query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
            # shape: (batch, doc_max)
            document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        
        
        #
        # similarity matrix
        # -------------------------------------------------------

        # clip & pad  embeddings
        if self.unified_query_length < query_embeddings.shape[1]:
            query_embeddings = query_embeddings[:, :self.unified_query_length, :]
            query_by_doc_mask = query_by_doc_mask[:, :self.unified_query_length, :]
        elif self.unified_query_length > query_embeddings.shape[1]:
            query_embeddings = F.pad(query_embeddings, (0, 0, 0, self.unified_query_length - query_embeddings.shape[1]))
            query_by_doc_mask = F.pad(query_by_doc_mask, (0, 0, 0, self.unified_query_length - query_by_doc_mask.shape[1]))

        if self.unified_document_length < query_embeddings.shape[1]:
            document_embeddings = document_embeddings[:, :self.unified_document_length, :]
            query_by_doc_mask = query_by_doc_mask[:, :, :self.unified_query_length]
        elif self.unified_document_length > document_embeddings.shape[1]:
            document_embeddings = F.pad(document_embeddings, (0, 0, 0, 
                                                              self.unified_document_length - document_embeddings.shape[1]))
            query_by_doc_mask = F.pad(query_by_doc_mask, (0, self.unified_document_length - query_by_doc_mask.shape[2]))
            
        # create sim matrix
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        # shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = cosine_matrix[:,None,:,:]
        query_by_doc_mask = query_by_doc_mask[:,None,:,:]
        
        #
        # duplicate cosine_matrix -> n-gram convolutions, then top-k pooling
        # ----------------------------------------------
        conv_results = []
        conv_results.append(torch.topk(cosine_matrix.squeeze(), k=self.kmax_pooling_size,sorted=True)[0])

        for conv in self.convolutions:
            cr = conv(cosine_matrix)
            cr_masked = cr * query_by_doc_mask
            cr_kmax_result = torch.topk(cr_masked.squeeze(), k=self.kmax_pooling_size, sorted=True)[0]
            conv_results.append(cr_kmax_result)

        #
        # flatten all paths together
        # -------------------------------------------------------
        
        per_query_results = torch.cat(conv_results, dim=-1)

        # add idf
        qry_idf = self.idf_lookup(query['tokens'])
        qry_idf = qry_idf.unsqueeze(-1)
        if self.unified_query_length <= qry_idf.shape[1]:
            qry_idf = qry_idf[:,:self.unified_query_length]
        elif self.unified_query_length > qry_idf.shape[1]:
            qry_idf = torch.nn.functional.pad(qry_idf, (0, 0, 0, self.unified_query_length - qry_idf.shape[1]))
        if self.unified_query_length <= query_pad_oov_mask.shape[1]:
            query_pad_oov_mask_unifiedlength = query_pad_oov_mask[:,:self.unified_query_length]
        elif self.unified_query_length > query_pad_oov_mask.shape[1]:
            query_pad_oov_mask_unifiedlength = torch.nn.functional.pad(
                query_pad_oov_mask, (0, self.unified_query_length - query_pad_oov_mask.shape[1]))
        query_pad_oov_mask_unifiedlength = query_pad_oov_mask_unifiedlength.unsqueeze(-1)
        
        qry_idf_softmax = utils.masked_softmax(qry_idf, query_pad_oov_mask_unifiedlength, dim=1)
        per_query_withsalc_results = torch.cat((per_query_results, qry_idf_softmax), dim=-1)
        
        # flatting
        all_flat = per_query_withsalc_results.view(per_query_withsalc_results.shape[0], -1)

        #
        # dense layer
        # -------------------------------------------------------

        dense_out = F.relu(self.dense(all_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)

        output = torch.squeeze(dense_out, 1)
        return {"rels":output}

