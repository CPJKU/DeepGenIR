from typing import Dict, Iterator, List
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          


class Conv_KNRM(Model):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    third-hand reference: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py (tensorflow)
    https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/baselines/CKNRM.py (pytorch)

    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 n_kernels: int,
                 conv_out_dim:int) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = torch.tensor(self.kernel_mus(n_kernels), 
                               dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = torch.tensor(self.kernel_sigmas(n_kernels),
                                  dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels)
        
        # stride is always 1, = the size of the steps for each sliding step
        self.conv_1 = nn.Sequential(
            nn.Conv1d(kernel_size = 1, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())

        self.conv_2 = nn.Sequential(
            nn.Conv1d(kernel_size = 2, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())

        self.conv_3 = nn.Sequential(
            nn.Conv1d(kernel_size = 3, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * 9, 1, bias=False) 

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 1).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 1).float()

        # shape: (batch, query_max)
        query_pad_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_mask = (document["tokens"] > 0).float()


        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        #
        # 1-3 gram convolutions for query & document
        # -------------------------------------------------------

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2) # doesn't need padding because kernel_size = 1
        query_embeddings_t_p2 = nn.functional.pad(query_embeddings_t,(0,1)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last column)
        query_embeddings_t_p3 = nn.functional.pad(query_embeddings_t,(0,2)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last 2 columns)

        document_embeddings_t = document_embeddings.transpose(1, 2) # doesn't need padding because kernel_size = 1
        document_embeddings_t_p2 = nn.functional.pad(document_embeddings_t,(0,1)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last column)
        document_embeddings_t_p3 = nn.functional.pad(document_embeddings_t,(0,2)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last 2 columns)

        query_conv_1 = self.conv_1(query_embeddings_t).transpose(1, 2) 
        query_conv_2 = self.conv_2(query_embeddings_t_p2).transpose(1, 2)
        query_conv_3 = self.conv_3(query_embeddings_t_p3).transpose(1, 2)

        document_conv_1 = self.conv_1(document_embeddings_t).transpose(1, 2) 
        document_conv_2 = self.conv_2(document_embeddings_t_p2).transpose(1, 2)
        document_conv_3 = self.conv_3(document_embeddings_t_p3).transpose(1, 2)

        #
        # similarity matrix & gaussian kernels & soft TF for all conv combinations
        # -------------------------------------------------------

        # TODO question about the query_by_doc_mask - shouldn't we remove the last & 2-nd last "1" in every sample - row based on the conv, because l_out is < l_in so we leave the last element wrongly
        sim_q1_d1 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q1_d2 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q1_d3 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_3, query_by_doc_mask, query_pad_mask)
 
        sim_q2_d1 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q2_d2 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q2_d3 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_3, query_by_doc_mask, query_pad_mask)
 
        sim_q3_d1 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q3_d2 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q3_d3 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_3, query_by_doc_mask, query_pad_mask)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat([
            sim_q1_d1,sim_q1_d2,sim_q1_d3,
            sim_q2_d1,sim_q2_d2,sim_q2_d3,
            sim_q3_d1,sim_q3_d2,sim_q3_d3],1)

        dense_out = self.dense(all_grams)
        tanh_out = torch.tanh(dense_out)

        output = torch.squeeze(tanh_out, 1)
        
        return {"rels":output}
 
    ## create a match matrix between query & document terms
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)         
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        return per_kernel

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
    
    def cuda(self, device=None):
        self = super().cuda(device)
        self.mu = self.mu.cuda(device)
        self.sigma = self.sigma.cuda(device)
        return self 

