from typing import Dict, List, Tuple, Optional, overload
import numpy as np
import time
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.nn import TransformerEncoderLayer, TransformerEncoder

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

import torch.distributions as distributions

class SkipRNN(torch.nn.Module):    
    
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(SkipRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = False
    
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return input, input[-1]
    
    
class TransformerPointerGeneratorNetwork(Model):
    '''
        Based on our PyTorch implementation of PGN for IR 
        which in turn is based on the PGN model implemented
        in Summarus by Ilya Gusev:
        https://github.com/IlyaGusev/summarus
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 lstm_hidden_dim: int = 300,
                 enc_lstm_num_layers: int = 1,
                 enc_transf_num_layers: int = 2,
                 enc_transf_intermediate_size: int = 512,
                 enc_transf_num_heads: int = 2, # num of heads in multiheadattention
                 enc_transf_dropout: float  = 0.1,
                 projection_dim: int = None,
                 tie: bool = True) -> None:
        super(TransformerPointerGeneratorNetwork, self).__init__(vocab)

        self.target_namespace = "tokens"
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self.target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self.target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, self.target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(self.target_namespace)
        self.enc_lstm_num_layers = enc_lstm_num_layers
        
        # Encoder
        self._source_embedder = source_embedder
        embedding_dim = self._source_embedder.get_output_dim()
        encoder_layer = TransformerEncoderLayer(embedding_dim, enc_transf_num_heads, 
                                                enc_transf_intermediate_size, enc_transf_dropout)
        self._pos_encoder = PositionalEncoding(embedding_dim, dropout=0)
        self._transf_encoder_pgn = TransformerEncoder(encoder_layer, enc_transf_num_layers)


        if enc_lstm_num_layers > 0:
            self._encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=embedding_dim, 
                                                                hidden_size=lstm_hidden_dim, 
                                                                num_layers = enc_lstm_num_layers,
                                                                batch_first=True))
        else:
            self._encoder = PytorchSeq2SeqWrapper(SkipRNN(input_size=embedding_dim, 
                                                          hidden_size=lstm_hidden_dim))
        self._encoder_output_dim = self._encoder.get_output_dim()

        # Decoder
        _target_embedding_dim = source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(self.target_namespace)
        self._target_embedder = source_embedder # making common embedding here

        self._decoder_output_dim = self._encoder_output_dim
        self._decoder_cell = LSTMCell(_target_embedding_dim, self._decoder_output_dim)

        if projection_dim != 'None':
            self._projection_dim = projection_dim
        else:
            self._projection_dim = self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(self._decoder_output_dim + lstm_hidden_dim, self._projection_dim)
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
        
        self._p_gen_layer = Linear(self._decoder_output_dim * 2 + embedding_dim, 1)
        self._attention = AdditiveAttention(self._decoder_output_dim, self._encoder_output_dim)

        # parameters
        self._eps = 1e-31
        
    def forward(self,
                source: Dict[str, torch.LongTensor],
                source_local_token_ids: torch.Tensor,
                target: Dict[str, torch.LongTensor],
                target_local_token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:

        enc_start_time = time.time()
        state = self._encode(source)
        enc_time = time.time() - enc_start_time

        target_token_ids = target[self.target_namespace].long()
        source_token_ids = source[self.target_namespace]
        
        extra_zeros, modified_source_tokens, modified_target_tokens = self._prepare(
            source_token_ids, source_local_token_ids, target_token_ids, target_local_token_ids)

        state["source_tokens"] = modified_source_tokens
        state["target_tokens"] = modified_target_tokens
        state["extra_zeros"] = extra_zeros
        
        state = self._init_decoder_state(state)
        forward_loop_res = self._forward_loop(state, target)
        
        logprobs = forward_loop_res["logprobs"]
        uncertainty = forward_loop_res["uncertainty"]
        
        return {"rels":logprobs, "enc_time":enc_time, "uncertainty": uncertainty}

    def _prepare(self,
                 source_token_ids: torch.LongTensor,
                 source_local_token_ids: torch.Tensor,
                 target_token_ids: torch.LongTensor = None,
                 target_local_token_ids: torch.Tensor = None):
        
        batch_size = source_token_ids.size(0)
        source_max_length = source_token_ids.size(1)

        # Concat target tokens
        token_ids = torch.cat((source_token_ids, target_token_ids), 1)
        local_token_ids = torch.cat((source_local_token_ids.long(), target_local_token_ids.long()), 1)

        is_unk = torch.eq(token_ids, self._unk_index).long()
        # Create tensor with ids of unknown tokens only.
        # Those ids are batch-local.
        unk_only = local_token_ids * is_unk
        
        # Recalculate batch-local ids to range [1, count_of_unique_unk_tokens].
        # All known tokens have zero id.
        unk_token_nums = local_token_ids.new_zeros((batch_size, local_token_ids.size(1)))
        for i in range(batch_size):
            unique = torch.unique(unk_only[i, :], return_inverse=True, sorted=True)[1]
            unk_token_nums[i, :] = unique

        # Replace DEFAULT_OOV_TOKEN id with new batch-local ids starting from vocab_size
        # For example, if vocabulary size is 50000, the first unique unknown token will have 50000 index,
        # the second will have 50001 index and so on.
        token_ids = token_ids - token_ids * is_unk + (self._vocab_size - 1) * is_unk + unk_token_nums

        # Remove target unknown tokens that do not exist in source tokens
        max_source_num = torch.max(token_ids[:, :source_max_length], dim=1)[0]
        vocab_size = max_source_num.new_full((1,), self._vocab_size-1)
        max_source_num = torch.max(max_source_num, other=vocab_size).unsqueeze(1).expand((-1, token_ids.size(1)))
        unk_target_tokens_mask = torch.gt(token_ids, max_source_num).long()
        token_ids = token_ids - token_ids * unk_target_tokens_mask + self._unk_index * unk_target_tokens_mask
        modified_target_tokens = token_ids[:, source_max_length:]
        modified_source_tokens = token_ids[:, :source_max_length]

        # Count unique unknown source tokens to create enough zeros for final distribution
        source_unk_count = torch.max(unk_token_nums[:, :source_max_length])
        extra_zeros = token_ids.new_zeros((batch_size, source_unk_count), dtype=torch.float32)
        
        return extra_zeros, modified_source_tokens, modified_target_tokens

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (max_input_sequence_length, batch_size, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens).transpose(0, 1)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        
        embedded_input = self._pos_encoder(embedded_input)
        embedded_input_contextualized = self._transf_encoder_pgn(embedded_input, src_key_padding_mask = ~util.get_text_field_mask(source_tokens).bool()).transpose(0, 1)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder.forward(embedded_input_contextualized, source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional())
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output

        encoder_outputs = state["encoder_outputs"]
        state["decoder_context"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size(0)

        targets = target_tokens["tokens"] # <-- IN for our TARGET EMBEDDER ##
        _, target_sequence_length = targets.size()

        num_decoding_steps = target_sequence_length - 1

        step_proba: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            input_choices = targets[:, timestep]
            output_projections, state = self._prepare_output_projections(input_choices, state)
            final_dist = self._get_final_dist(state, output_projections)
            step_proba.append(final_dist)
        
        num_classes = step_proba[0].size(1)
        proba = step_proba[0].new_zeros((batch_size, num_classes, len(step_proba)))
        for i, p in enumerate(step_proba):
            proba[:, :, i] = p
        
        logprobs = self.get_logprobs(proba, state["target_tokens"])
        
        uncertainty = []
        if not self.training:
            uncertainty = self.get_by_word_uncertainty(proba)
        
        return {"logprobs" : logprobs, "uncertainty" : uncertainty}
    
    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]

        is_unk = (last_predictions >= self._vocab_size).long()
        last_predictions_fixed = last_predictions - last_predictions * is_unk + self._unk_index * is_unk

        decoder_embedded_input = self._target_embedder.forward({"tokens":last_predictions_fixed})
        decoder_hidden, decoder_context = self._decoder_cell(decoder_embedded_input, (decoder_hidden, decoder_context))

        attn_scores = self._attention.forward(decoder_hidden, encoder_outputs, source_mask)
        attn_context = util.weighted_sum(encoder_outputs, attn_scores)
        
        decoder_output = torch.cat((attn_context, decoder_hidden), -1)

        output_projections = self._output_projection_layer(self._hidden_projection_layer(decoder_output))

        state["decoder_input"] = decoder_embedded_input
        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_scores"] = attn_scores
        state["attn_context"] = attn_context

        return output_projections, state

    def _get_final_dist(self, state: Dict[str, torch.Tensor], output_projections):
        attn_dist = state["attn_scores"]
        source_tokens = state["source_tokens"]
        extra_zeros = state["extra_zeros"]
        attn_context = state["attn_context"]
        decoder_input = state["decoder_input"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]

        #decoder_state = torch.cat((decoder_hidden, decoder_context), 1)
        p_gen = self._p_gen_layer(torch.cat((attn_context, decoder_hidden, decoder_input), 1))
        p_gen = torch.sigmoid(p_gen)

        vocab_dist = F.softmax(output_projections, dim=-1)

        vocab_dist = vocab_dist * p_gen
        attn_dist = attn_dist * (1.0 - p_gen)
        if extra_zeros.size(1) != 0:
            vocab_dist = torch.cat((vocab_dist, extra_zeros), 1)
        final_dist = vocab_dist.scatter_add(1, source_tokens, attn_dist)
        normalization_factor = final_dist.sum(1, keepdim=True)
        final_dist = final_dist / normalization_factor 

        return final_dist

    def get_logprobs(self, proba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
        logproba = torch.log(proba + self._eps)
        targets = targets[:, 1:]
        rels = torch.nn.NLLLoss(ignore_index=0, reduction='none')(logproba, targets)
        return -torch.sum(rels, dim=1)
    
    def get_by_word_uncertainty(self, proba: torch.LongTensor) -> torch.Tensor:
        # convert probabilities to uncertainty values
        res = torch.zeros([proba.shape[0], proba.shape[2]], requires_grad=False)
        for q in range(proba.shape[0]): #batch
            for qw in range(proba.shape[2]): #query
                # getting the prob distribution over all words in the vocab for
                # the current word-position and calculating entropy
                word_n_dist = self.nucleus_sampling_entropy_prep(proba[q,:,qw], 0.95)
                res[q,qw] = distributions.Categorical(probs=word_n_dist).entropy()
        
        return res
    
    def nucleus_sampling_entropy_prep (self, probs, top_p=0.95):
        # Nucleus Sampling:
        # paper: THE CURIOUS CASE OF NEURAL TEXT DeGENERATION
        # [Holtzman et. al] https://arxiv.org/pdf/1904.09751.pdf
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        top_idx = len(cumulative_probs[cumulative_probs < top_p]) + 1 # index of cumulative sum equal or greater than top_p
        
        new_dist = sorted_probs[:top_idx] / sorted_probs[:top_idx].sum() # normalization on top_p
        
        return new_dist
    
    