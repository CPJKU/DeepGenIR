import numpy as np
from typing import Dict, List, Tuple, Optional, overload
import math
import copy
import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Attention
from allennlp.nn import util
from allennlp.modules.attention.additive_attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from models.positional_encoding import PositionalEncoding

class SeqToSeqTransformers(Model):
    def __init__(self,
                 vocab: Vocabulary, # vocab, size of vocab
                 source_embedder: TextFieldEmbedder, 
                 encoder_num_layers: int = 2,
                 decoder_num_layers: int = 2,
                 transf_intermediate_size: int = 512,
                 num_heads: int = 2, # num of heads in multiheadattention
                 dropout: float  = 0.1,
                 projection_dim: int = None,
                 tie: bool = True) -> None:
        super(SeqToSeqTransformers, self).__init__(vocab) # (?)

        target_namespace = "tokens"
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(target_namespace)
        
        # Encoder 
        self._source_embedder = source_embedder
        embedding_dim = self._source_embedder.get_output_dim() # embedding_dim == ninp 
        self._pos_encoder = PositionalEncoding(embedding_dim, dropout=0)

        if encoder_num_layers > 0:
            encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, transf_intermediate_size, dropout)
            self._transf_encoder = TransformerEncoder(encoder_layer, encoder_num_layers)
        else:
            raise ConfigurationError("Minimum 1 level of transformers required.. at this point")
        # encoder output dim in case of transformers == embedding dimension
        self._encoder_output_dim = embedding_dim
        # for later: add Positional Encoding
        # self._pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Decoder
        target_embedding_dim = source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._target_embedder = source_embedder # making common embedding here
        
        self._pos_decoder = PositionalEncoding(target_embedding_dim, dropout=0)

        if decoder_num_layers > 0:
            decoder_layer = TransformerDecoderLayer(target_embedding_dim, num_heads, transf_intermediate_size, dropout)
            self._transf_decoder = TransformerDecoder(decoder_layer, decoder_num_layers)
        else:
            raise ConfigurationError("Minimum 1 level of transformers required")


        self._decoder_output_dim = self._encoder_output_dim
        
        if projection_dim != 'None':
            self._projection_dim = projection_dim
        else:
            self._projection_dim = self._source_embedder.get_output_dim()
        # self._hidden_projection_layer = Linear(self._decoder_output_dim + hidden_dim, self._projection_dim)
        self._output_projection_layer = Linear(self._projection_dim, self._num_classes)
        if tie:
            if not isinstance(self._target_embedder, BasicTextFieldEmbedder):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Source text embedder is not an instance of `BasicTextFieldEmbedder`.")
            _embedder = self._target_embedder._token_embedders['tokens']
            if not isinstance(_embedder, Embedding):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Selected source embedder is not an instance of `Embedding`.")
            _embedder_weight_shape = _embedder.weight.shape
            _output_projection_weight_shape = self._output_projection_layer.weight.shape
            if  _embedder_weight_shape != _output_projection_weight_shape :
                raise ConfigurationError(f"Unable to tie embeddings due to mismatch in shared weights."
                                         f"source  -  {_embedder_weight_shape}"
                                         f"target  -  {_output_projection_weight_shape}")
            self._output_projection_layer.weight = _embedder.weight

        # parameters
        # self._max_decoding_steps = max_decoding_steps
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        enc_start_time = time.time()
        state = self._encode(document)
        enc_time = time.time() - enc_start_time

        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]
                
        logprobs, softmax_time, dec_time = self._decode(state, query)

        return {"rels":logprobs, "enc_time":enc_time, "softmax_time":softmax_time, "dec_time":dec_time}

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens).transpose(0, 1)
        embedded_input = self._pos_encoder(embedded_input)
        
        # shape: (batch_size, max_input_sequence_length)
        # pdb.set_trace()
        source_mask = ~util.get_text_field_mask(source_tokens).bool()
        
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._transf_encoder(embedded_input, src_key_padding_mask = source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _decode(self, state: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_padding_mask = state["source_mask"]
        tgt_padding_mask = ~util.get_text_field_mask(target_tokens).bool()

        batch_size = source_padding_mask.size(0)

        
        # pdb.set_trace()
        #tag_vec = tgt_padding_mask[0].clone().detach().requires_grad_(False).float()
        #tag_vec[:] = 1
        #diag = torch.diag(tag_vec[:-1], diagonal=1)
        #mask = diag.masked_fill(diag==1, float('-inf'))
        ####mask = mask.masked_fill(mask==0, float(1.0))
        
        sz = tgt_padding_mask.size(1)
        tag_vec = tgt_padding_mask.new_ones(sz, sz).detach().requires_grad_(False).float()
        mask = (torch.triu(tag_vec) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        dec_start_time = time.time()
        embedded_targets = self._target_embedder.forward(target_tokens).transpose(0, 1)
        embedded_targets = self._pos_decoder(embedded_targets)
        
        decoder_outputs = self._transf_decoder(tgt = embedded_targets,
                                               memory = state["encoder_outputs"],
                                               memory_key_padding_mask = source_padding_mask,
                                               tgt_key_padding_mask = tgt_padding_mask,
                                               tgt_mask = mask) #MASK TGT WITH tgt_mask = *
        dec_time = time.time() - dec_start_time

        softmax_start_time = time.time()
        output_projections = self._output_projection_layer(decoder_outputs)
        final_dists = torch.nn.LogSoftmax(dim=-1)(output_projections) # ?
        logproba = final_dists.transpose(1, 2)
        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        softmax_time = time.time() - softmax_start_time

        return logprobs, softmax_time, dec_time
        
    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
        _targets = targets[:, 1:]#.transpose(dim0=0, dim1=1)
        _logproba = logproba[:-1, :, :].transpose(0, 2)
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba, _targets)
        return torch.sum(logprobs, dim=1)
    


