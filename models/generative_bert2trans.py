import numpy as np
from typing import Dict, List, Tuple, Optional, overload
import math
import copy
import time

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model
from allennlp.nn import util

from transformers import BertModel, BertConfig, AutoConfig, AutoModelForCausalLM

from models.positional_encoding import PositionalEncoding

import torch.distributions as distributions



class BERT2Transformer(Model):
    def __init__(self,
                 encoder_bert: BertModel, 
                 decoder_hidden_size: int = 128,
                 decoder_num_layers: int = 2, # num of transformer-encoder and -decoder layers
                 decoder_intermediate_size: int = 512,
                 decoder_num_heads: int = 2, # num of heads in multiheadattention
                 decoder_dropout: float  = 0.1,
                 tie_encoder_decoder: bool = False,
                 tie_decoder_output: bool = False) -> None:
        super(BERT2Transformer, self).__init__(vocab=None) # (?)

        # Encoder
        self.encoder_bert = encoder_bert
        self.vocab_size = self.encoder_bert.config.vocab_size
        self.num_classes = self.vocab_size
        _encoder_embedding_dim = self.encoder_bert.config.hidden_size 
        
        # Decoder
        self.tgt_embeddings = nn.Embedding(self.vocab_size, decoder_hidden_size, padding_idx=0)
        self.pos_decoder = PositionalEncoding(decoder_hidden_size, dropout=0)

        if tie_encoder_decoder:
            self.tgt_embeddings.weight = self.encoder_bert.embeddings.word_embeddings.weight
        
        decoder_layer = TransformerDecoderLayer(decoder_hidden_size, decoder_num_heads, 
                                                decoder_intermediate_size, decoder_dropout)
        self.transf_decoder = TransformerDecoder(decoder_layer, decoder_num_layers)
        
        # output projection
        self.output_projection_layer = Linear(decoder_hidden_size, self.num_classes)
        if tie_decoder_output:
            _tgt_shape = self.tgt_embeddings.weight.shape
            _output_shape = self.output_projection_layer.weight.shape
            if  _tgt_shape != _output_shape :
                raise ConfigurationError("Mismatch in shared weights; source: %s target: %s" % (str(_tgt_shape), 
                                                                                                str(_output_shape)))
            self.output_projection_layer.weight = self.tgt_embeddings.weight
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_start_time = time.time()
        state = self._encode(document)
        enc_time = time.time() - enc_start_time
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]

        logprobs, softmax_time, dec_time, uncertainty = self._decode(state, query)

        return {"rels":logprobs, "uncertainty":uncertainty, "enc_time":enc_time, "softmax_time":softmax_time, "dec_time":dec_time}

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # source_tokens["tokens"] shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens) # 1 - not masked, 0 - masked;
        segs = source_mask.new_zeros(source_mask.size()).detach().requires_grad_(False)
        #pdb.set_trace()
        ctx_embedded_inputs = self.encoder_bert(input_ids=source_tokens["tokens"], token_type_ids=segs, attention_mask=source_mask, return_dict=True)
        
        source_mask = ~source_mask.bool() # casting mask back into format for vanilla transformers;
        
        return {
                "source_mask": source_mask,
                "encoder_outputs": ctx_embedded_inputs['last_hidden_state']
        }

    def _decode(self, state: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_padding_mask = state["source_mask"]
        tgt_padding_mask = ~util.get_text_field_mask(target_tokens).bool()

        batch_size = source_padding_mask.size(0)
        
        sz = tgt_padding_mask.size(1)
        tag_vec = tgt_padding_mask.new_ones(sz, sz).detach().requires_grad_(False).float()
        mask = (torch.triu(tag_vec) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        embedded_targets = self.tgt_embeddings.forward(target_tokens["tokens"]).transpose(0, 1) # (len, batch, nfeat)
        embedded_targets = self.pos_decoder(embedded_targets)
        
        dec_start_time = time.time()
        decoder_outputs = self.transf_decoder(tgt = embedded_targets,
                                              memory = state["encoder_outputs"].transpose(0, 1),
                                              memory_key_padding_mask = source_padding_mask,
                                              tgt_key_padding_mask = tgt_padding_mask,
                                              tgt_mask = mask) #MASK TGT WITH tgt_mask = *
        dec_time = time.time() - dec_start_time
        
        softmax_start_time = time.time()
        output_projections = self.output_projection_layer(decoder_outputs.transpose(0, 1)) #(batch, len, num_class)
        
        uncertainty = []
        if not self.training:
            uncertainty = self.get_uncertainty_by_logits(output_projections)
        
        logproba = torch.nn.LogSoftmax(dim=-1)(output_projections)
        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        softmax_time = time.time() - softmax_start_time

        return logprobs, softmax_time, dec_time, uncertainty

    def get_uncertainty_by_logits(self, logits: torch.LongTensor) -> torch.Tensor:
        
        by_word_uncertainty = distributions.Categorical(logits=logits).entropy() #logits or probs?
        #
        if torch.isnan(by_word_uncertainty).sum() > 0 :
            print("Nan entropy!")
            raise ValueError
        #
        # by_query_uncertainty = by_word_uncertainty.mean(dim = 1)
        return by_word_uncertainty
    
    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
        _targets = targets[:, 1:]
        _logproba = logproba[:, :-1, :].transpose(1, 2)
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba, _targets)
        return torch.sum(logprobs, dim=1)

    
'''
# encoder: bert
# decoder: first uses encoder_bert to contexualize and then decoder_num_layers of cross attention
class BERT2TransformerVariation1(Model):
    def __init__(self,
                 encoder_bert: BertModel, 
                 decoder_hidden_size: int = 128,
                 decoder_num_layers: int = 1, # num of transformer-encoder and -decoder layers
                 decoder_intermediate_size: int = 512,
                 decoder_num_heads: int = 2, # num of heads in multiheadattention
                 decoder_dropout: float  = 0.1,
                 tie_decoder_output: bool = False) -> None:
        super(BERT2TransformerVariation1, self).__init__(vocab=None) # (?)

        # Encoder
        self.encoder_bert = encoder_bert
        self.vocab_size = self.encoder_bert.config.vocab_size
        self.num_classes = self.vocab_size
        _encoder_embedding_dim = self.encoder_bert.config.hidden_size 
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(decoder_hidden_size, decoder_num_heads, 
                                                decoder_intermediate_size, decoder_dropout)
        self.transf_decoder = TransformerDecoder(decoder_layer, decoder_num_layers)
        
        # output projection
        self.output_projection_layer = Linear(decoder_hidden_size, self.num_classes)
        if tie_decoder_output:
            _tgt_shape = self.encoder_bert.embeddings.word_embeddings.weight
            _output_shape = self.output_projection_layer.weight.shape
            if  _tgt_shape != _output_shape :
                raise ConfigurationError("Mismatch in shared weights; source: %s target: %s" % (str(_tgt_shape), 
                                                                                                str(_output_shape)))
            self.output_projection_layer.weight = self.encoder_bert.embeddings.word_embeddings.weight
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_start_time = time.time()
        state = self.encode(document)
        enc_time = time.time() - enc_start_time
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]

        logprobs, softmax_time, dec_time = self.decode(state, query)

        return {"rels":logprobs, "enc_time":enc_time, "softmax_time":softmax_time, "dec_time":dec_time}

    def encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # source_tokens["tokens"] shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens) # 1 - not masked, 0 - masked;
        segs = source_mask.new_zeros(source_mask.size()).detach().requires_grad_(False)

        ctx_embedded_inputs = self.encoder_bert(input_ids=source_tokens["tokens"], token_type_ids=segs, attention_mask=source_mask, return_dict=True)
        
        source_mask = ~source_mask.bool() # casting mask back into format for vanilla transformers;
        
        return {
                "source_mask": source_mask,
                "encoder_outputs": ctx_embedded_inputs['last_hidden_state']
        }

    def decode(self, state: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        
        # contexualizing target
        target_mask = util.get_text_field_mask(target_tokens) # 1 - not masked, 0 - masked;
        segs = target_mask.new_zeros(target_mask.size()).detach().requires_grad_(False)
        ctx_embedded_targets = self.encoder_bert(input_ids=target_tokens["tokens"], token_type_ids=segs, attention_mask=target_mask, return_dict=True)['last_hidden_state']
        
        # shape: (batch_size, max_input_sequence_length)
        source_padding_mask = state["source_mask"]
        tgt_padding_mask = ~target_mask.bool()

        batch_size = source_padding_mask.size(0)
        
        sz = tgt_padding_mask.size(1)
        tag_vec = tgt_padding_mask.new_ones(sz, sz).detach().requires_grad_(False).float()
        mask = (torch.triu(tag_vec) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        dec_start_time = time.time()
        decoder_outputs = self.transf_decoder(tgt = ctx_embedded_targets.transpose(0, 1), # (len, batch, nfeat)
                                              memory = state["encoder_outputs"].transpose(0, 1),
                                              memory_key_padding_mask = source_padding_mask,
                                              tgt_key_padding_mask = tgt_padding_mask,
                                              tgt_mask = mask) #MASK TGT WITH tgt_mask = *
        dec_time = time.time() - dec_start_time

        softmax_start_time = time.time()
        output_projections = self.output_projection_layer(decoder_outputs.transpose(0, 1)) #(batch, len, num_class)
        logproba = torch.nn.LogSoftmax(dim=-1)(output_projections)
        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        softmax_time = time.time() - softmax_start_time

        return logprobs, softmax_time, dec_time

    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
        _targets = targets[:, 1:]
        _logproba = logproba[:, :-1, :].transpose(1, 2)
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba, _targets)
        return torch.sum(logprobs, dim=1)




### WIP seq2seq with BERT
class BERT2BERT(Model):
    def __init__(self,
                 encoder_bert: BertModel, 
                 decoder_bert_id: str = "random",
                 decoder_hidden_size: int = 128,
                 decoder_num_layers: int = 2, # num of transformer-encoder and -decoder layers
                 decoder_intermediate_size: int = 512,
                 decoder_num_heads: int = 2, # num of heads in multiheadattention
                 decoder_dropout: float  = 0.1,
                 tie_encoder_decoder: bool = False,
                 tie_decoder_output: bool = False) -> None:
        super(BERT2BERT, self).__init__(vocab=None) # (?)

        # Encoder
        self.encoder_bert = encoder_bert
        self.vocab_size = self.encoder_bert.config.vocab_size
        self.num_classes = self.vocab_size
        _encoder_embedding_dim = self.encoder_bert.config.hidden_size 
                
        # Decoder
        if decoder_bert_id == "random":
            _bert_decoder_config = BertConfig(vocab_size = self.encoder_bert.config.vocab_size,
                                              hidden_size = decoder_hidden_size,
                                              num_hidden_layers = decoder_num_layers,
                                              num_attention_heads = decoder_num_heads,
                                              intermediate_size = decoder_intermediate_size,
                                              hidden_dropout_prob = decoder_dropout,
                                              attention_probs_dropout_prob = decoder_dropout,
                                             is_decoder = True,
                                             add_cross_attention = True)
            self.decoder_bert = AutoModelForCausalLM.from_config(_bert_decoder_config)
            pdb.set_trace()
        else:
            #loading config through loading the model
            _bert_decoder_config = BertModel.from_pretrained(decoder_bert_id, cache_dir="cache").config 
            _bert_decoder_config.is_decoder = True
            _bert_decoder_config.add_cross_attention = True
            self.decoder_bert = AutoModelForCausalLM.from_pretrained(decoder_bert_id, cache_dir='cache', 
                                                                     config=_bert_decoder_config) 
        
        if tie_encoder_decoder:
            self.decoder_bert.bert.embeddings.word_embeddings.weight = self.encoder_bert.embeddings.word_embeddings.weight
                
        # output projection
        if tie_decoder_output:
            _tgt_shape = self.decoder_bert.bert.embeddings.word_embeddings.weight.shape
            _output_shape = self.decoder_bert.cls.predictions.decoder.weight.shape
            if  _tgt_shape != _output_shape :
                raise ConfigurationError("Mismatch in shared weights; source: %s target: %s" % (str(_tgt_shape), 
                                                                                                str(_output_shape)))
            self.decoder_bert.cls.predictions.decoder.weight = self.decoder_bert.bert.embeddings.word_embeddings.weight
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_start_time = time.time()
        state = self.encode(document)
        enc_time = time.time() - enc_start_time
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]

        logprobs, softmax_time, dec_time = self.decode(state, query)

        return {"rels":logprobs, "enc_time":enc_time, "softmax_time":softmax_time, "dec_time":dec_time}

    def encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # source_tokens["tokens"] shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens) # 1 - not masked, 0 - masked;
        segs = source_mask.new_zeros(source_mask.size()).detach().requires_grad_(False)

        ctx_embedded_inputs = self.encoder_bert(input_ids=source_tokens["tokens"], token_type_ids=segs, attention_mask=source_mask, return_dict=True)
        
        return {
                "source_mask": source_mask,
                "encoder_outputs": ctx_embedded_inputs['last_hidden_state'],
        }

    def decode(self, state: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_padding_mask = state["source_mask"]
        tgt_padding_mask = util.get_text_field_mask(target_tokens) # 1 - not masked, 0 - masked;
        segs = tgt_padding_mask.new_zeros(tgt_padding_mask.size()).detach().requires_grad_(False)

        dec_start_time = time.time()
        decoder_outputs = self.decoder_bert(input_ids = target_tokens["tokens"],
                                            attention_mask = tgt_padding_mask,
                                            token_type_ids = segs,
                                            encoder_hidden_states = state["encoder_outputs"],
                                            encoder_attention_mask = source_padding_mask,
                                           return_dict=True)     
        dec_time = time.time() - dec_start_time

        softmax_start_time = time.time()
        logproba = torch.nn.LogSoftmax(dim=-1)(decoder_outputs.logits) # ?
        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        softmax_time = time.time() - softmax_start_time

        return logprobs, softmax_time, dec_time

    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
        _targets = targets[:, 1:]
        _logproba = logproba.transpose(1, 2)[:, :, :-1]
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba, _targets)
        return torch.sum(logprobs, dim=1)

'''

