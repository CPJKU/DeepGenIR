import numpy as np
from typing import Dict, List, Tuple, Optional, overload
import time

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

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


# EPS = 1e-20

class SkipRNN(torch.nn.Module):    
    
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(SkipRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = False
    
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return input, input[-1]
    
    
class SeqToSeqWithAttention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 lstm_hidden_dim: int = 300,
                 enc_lstm_num_layers: int = 1,
                 projection_dim: int = None,
                 tie: bool = True) -> None:
        super(SeqToSeqWithAttention, self).__init__(vocab)

        target_namespace = "tokens"
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(target_namespace)
        self.enc_lstm_num_layers = enc_lstm_num_layers
        
        # Encoder
        self._source_embedder = source_embedder
        embedding_dim = self._source_embedder.get_output_dim()
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
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
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
        
        self._attention = AdditiveAttention(self._decoder_output_dim, self._encoder_output_dim)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_start_time = time.time()
        state = self._encode(document)
        enc_time = time.time() - enc_start_time
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]
                
        state = self._init_decoder_state(state)
        logprobs = self._forward_loop(state, query)

        return {"rels":logprobs, "enc_time":enc_time}

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder.forward(embedded_input, source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        
        encoder_outputs = state["encoder_outputs"]
        
        if self.enc_lstm_num_layers > 0:
            final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                                 state["source_mask"],
                                                                 self._encoder.is_bidirectional())
            # Initialize the decoder hidden state with the final output of the encoder.
            # shape: (batch_size, decoder_output_dim)
            state["decoder_hidden"] = final_encoder_output
        else:
            state["decoder_hidden"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
            
        state["decoder_context"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size(0)
        
        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"] # <-- IN for our TARGET EMBEDDER ##
        _, target_sequence_length = targets.size()

        num_decoding_steps = target_sequence_length - 1

        step_logproba: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            target_token_timestep = targets[:, timestep]
            output_projections, state = self._prepare_output_projections(target_token_timestep, state)
            final_dist = torch.nn.LogSoftmax(dim=-1)(output_projections)
            step_logproba.append(final_dist)
            #final_dist = torch.nn.Softmax(dim=-1)(output_projections) # torch.nn.LogSoftmax(dim=-1)(output_projections)
            #if unlikely:
            #    final_dist = 1 - final_dist
            #step_logproba.append(torch.log(final_dist + EPS))
            
        num_classes = step_logproba[0].size(1)
        logproba = step_logproba[0].new_zeros((batch_size, num_classes, len(step_logproba)))
        for i, p in enumerate(step_logproba):
            logproba[:, :, i] = p

        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        
        return logprobs
    
    def _prepare_output_projections(self,
                                    target_token_timestep: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]

        decoder_embedded_input = self._target_embedder.forward({"tokens":target_token_timestep})
        decoder_hidden, decoder_context = self._decoder_cell(decoder_embedded_input, (decoder_hidden, decoder_context))

        attn_scores = self._attention.forward(decoder_hidden, encoder_outputs, source_mask)
        attn_context = util.weighted_sum(encoder_outputs, attn_scores)
        
        decoder_output = torch.cat((attn_context, decoder_hidden), -1)

        output_projections = self._output_projection_layer(self._hidden_projection_layer(decoder_output))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_scores"] = attn_scores
        state["attn_context"] = attn_context

        return output_projections, state

    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
                  
        targets = targets[:, 1:]
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(logproba, targets)
        return torch.sum(logprobs, dim=1)
        #return logprobs.sum(dim=1)/(logprobs!=0).sum(dim=1)

'''
class SeqToSeqWithAttentionBidirectional(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 max_decoding_steps: int=20,
                 hidden_dim: int = 300,
                 enc_lstm_num_layers: int = 1,
                 projection_dim: int = None,
                 tie: bool = True) -> None:
        super(SeqToSeqWithAttentionBidirectional, self).__init__(vocab)

        target_namespace = "tokens"
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(target_namespace)
        self.enc_lstm_num_layers = enc_lstm_num_layers
        
        # Encoder
        self._source_embedder = source_embedder
        embedding_dim = self._source_embedder.get_output_dim()
        if enc_lstm_num_layers > 0:
            self._encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=embedding_dim, 
                                                                hidden_size=hidden_dim, 
                                                                num_layers = enc_lstm_num_layers,
                                                                batch_first=True))
        else:
            self._encoder = PytorchSeq2SeqWrapper(SkipRNN(input_size=embedding_dim, 
                                                          hidden_size=hidden_dim))
        
        self._encoder_output_dim = self._encoder.get_output_dim()

        # Decoder
        _target_embedding_dim = source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._target_embedder = source_embedder # making common embedding here

        self._decoder_output_dim = self._encoder_output_dim
        self._decoder_cell_l2r = LSTMCell(_target_embedding_dim, self._decoder_output_dim)
        self._decoder_cell_r2l = LSTMCell(_target_embedding_dim, self._decoder_output_dim)

        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(2 * (self._decoder_output_dim + hidden_dim), self._projection_dim)
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
        
        self._attention = AdditiveAttention(self._decoder_output_dim, self._encoder_output_dim)

        # parameters
        self._max_decoding_steps = max_decoding_steps
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        state = self._encode(document)
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]
                
        state = self._init_decoder_state(state)
        logprobs = self._forward_loop(state, query)

        return logprobs

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder.forward(embedded_input, source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        
        encoder_outputs = state["encoder_outputs"]
        
        if self.enc_lstm_num_layers > 0:
            final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                                 state["source_mask"],
                                                                 self._encoder.is_bidirectional())
            # Initialize the decoder hidden state with the final output of the encoder.
            # shape: (batch_size, decoder_output_dim)
            state["decoder_hidden_l2r"] = final_encoder_output
            state["decoder_hidden_r2l"] = final_encoder_output
        else:
            state["decoder_hidden_l2r"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
            state["decoder_hidden_r2l"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
            
        state["decoder_context_l2r"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        state["decoder_context_r2l"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size(0)

        num_decoding_steps = self._max_decoding_steps
        
        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"] # <-- IN for our TARGET EMBEDDER ##
        _, target_sequence_length = targets.size()

        num_decoding_steps = target_sequence_length - 1

        finalembeds_step_l2r: List[torch.Tensor] = []
        finalembeds_step_r2l: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            target_token_timestep_l2r = targets[:, timestep]
            target_token_timestep_r2l = targets[:, num_decoding_steps - timestep]
            final_embed_l2r, final_embed_r2l, state = self._prepare_final_embeds(target_token_timestep_l2r, 
                                                                                 target_token_timestep_r2l, state)
            finalembeds_step_l2r.append(final_embed_l2r)
            finalembeds_step_r2l.append(final_embed_r2l)
        
        # we drop the last predictions of both sides (start and end are omitted), 
        # so the number of predictions is seq_len - 2
        finalembeds_step: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps - 1):
            finalembeds_step.append(torch.cat((finalembeds_step_l2r[timestep], 
                                               finalembeds_step_r2l[num_decoding_steps - 2 - timestep]), -1))
            
        finalembeds = finalembeds_step[0].new_zeros((batch_size, finalembeds_step[0].size(1), len(finalembeds_step)))
        for i, p in enumerate(finalembeds_step):
            finalembeds[:, :, i] = p

        output_projections = self._output_projection_layer(self._hidden_projection_layer(finalembeds.transpose(1, 2)))
        logproba = torch.nn.LogSoftmax(dim=-1)(output_projections)
        
        logprobs = self.get_logprobs(logproba, state["target_tokens"])
        
        return logprobs
    
    def _prepare_final_embeds(self, target_token_timestep_l2r: torch.Tensor,
                              target_token_timestep_r2l: torch.Tensor,
                              state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        decoder_hidden_l2r = state["decoder_hidden_l2r"]
        decoder_context_l2r = state["decoder_context_l2r"]
        decoder_hidden_r2l = state["decoder_hidden_r2l"]
        decoder_context_r2l = state["decoder_context_r2l"]

        decoder_embedded_input_l2r = self._target_embedder.forward({"tokens":target_token_timestep_l2r})
        decoder_embedded_input_r2l = self._target_embedder.forward({"tokens":target_token_timestep_r2l})
        
        decoder_hidden_l2r, decoder_context_l2r = self._decoder_cell_l2r(decoder_embedded_input_l2r, 
                                                                         (decoder_hidden_l2r, decoder_context_l2r))
        decoder_hidden_r2l, decoder_context_r2l = self._decoder_cell_r2l(decoder_embedded_input_r2l, 
                                                                         (decoder_hidden_r2l, decoder_context_r2l))
        
        attn_scores_l2r = self._attention.forward(decoder_hidden_l2r, encoder_outputs, source_mask)
        attn_context_l2r = util.weighted_sum(encoder_outputs, attn_scores_l2r)
        
        attn_scores_r2l = self._attention.forward(decoder_hidden_r2l, encoder_outputs, source_mask)
        attn_context_r2l = util.weighted_sum(encoder_outputs, attn_scores_r2l)
        
        final_embed_l2r = torch.cat((attn_context_l2r, decoder_hidden_l2r), -1)
        final_embed_r2l = torch.cat((attn_context_r2l, decoder_hidden_r2l), -1)

        state["decoder_hidden_l2r"] = decoder_hidden_l2r
        state["decoder_context_l2r"] = decoder_context_l2r
        state["decoder_hidden_r2l"] = decoder_hidden_r2l
        state["decoder_context_r2l"] = decoder_context_r2l
        
        return final_embed_l2r, final_embed_r2l, state

    def get_logprobs(self, logproba: torch.LongTensor,
                  targets: torch.LongTensor) -> torch.Tensor:
                  
        _targets = targets[:, 1:-1]
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(logproba.transpose(1, 2), _targets)
        return torch.sum(logprobs, dim=1)
        
        

class SeqToSeqWithAttentionFastDecoding(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 idfcf_path: str,
                 max_decoding_steps: int=20,
                 hidden_dim: int = 300,
                 enc_lstm_num_layers: int = 1,
                 projection_dim: int = None,
                 tie: bool = True,
                 output_mode_type="nce") -> None:
        super(SeqToSeqWithAttentionEfficient, self).__init__(vocab)

        target_namespace = "tokens"
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(target_namespace)
        self.enc_lstm_num_layers = enc_lstm_num_layers
        
        # Encoder
        self._source_embedder = source_embedder
        embedding_dim = self._source_embedder.get_output_dim()
        if enc_lstm_num_layers > 0:
            self._encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=embedding_dim, 
                                                                hidden_size=hidden_dim, 
                                                                num_layers = enc_lstm_num_layers,
                                                                batch_first=True))
        else:
            self._encoder = PytorchSeq2SeqWrapper(SkipRNN(input_size=embedding_dim, 
                                                          hidden_size=hidden_dim))
        
        self._encoder_output_dim = self._encoder.get_output_dim()

        # Decoder
        _target_embedding_dim = source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._target_embedder = source_embedder # making common embedding here

        self._decoder_output_dim = self._encoder_output_dim
        self._decoder_cell = LSTMCell(_target_embedding_dim, self._decoder_output_dim)

        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(self._decoder_output_dim + hidden_dim, self._projection_dim)
        
        noise_distibution = self.get_noise_tensor(idfcf_path, vocab)
        self._output_projection_layer = OutputLinear(hidden_dim=self._projection_dim,
                                                     num_classes=self._num_classes,
                                                     noise=noise_distibution,
                                                     mode_type=output_mode_type)
        if tie:
            if not isinstance(self._target_embedder, BasicTextFieldEmbedder):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Source text embedder is not an instance of `BasicTextFieldEmbedder`.")
            _embedder = self._target_embedder._token_embedders['tokens']
            if not isinstance(_embedder, Embedding):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Selected source embedder is not an instance of `Embedding`.")
            _embedder_weight_shape = _embedder.weight.shape
            _output_projection_weight_shape = self._output_projection_layer.emb.weight.shape
            if  _embedder_weight_shape != _output_projection_weight_shape :
                raise ConfigurationError(f"Unable to tie embeddings due to mismatch in shared weights."
                                         f"source  -  {_embedder_weight_shape}"
                                         f"target  -  {_output_projection_weight_shape}")
            self._output_projection_layer.emb.weight = _embedder.weight
        
        self._attention = AdditiveAttention(self._decoder_output_dim, self._encoder_output_dim)

        # parameters
        self._max_decoding_steps = max_decoding_steps
       
    def get_noise_tensor(self, idfcf_path, vocab):
        ## loading IDF values
        with open (idfcf_path, 'rb') as fr:
            idfcf_dic = pickle.load(fr)

        _cfs = []
        for v_i in range(vocab.get_vocab_size()):
            v = vocab.get_token_from_index(v_i)
            if v not in idfcf_dic:
                _cfs.append(1)
            else:
                _cfs.append(idfcf_dic[v][1])
        _cfs = np.array(_cfs)
        
        total = _cfs.sum()
        noise = _cfs / total
        assert abs(noise.sum() - 1) < 0.001
    
        _noise_tensor = nn.Parameter(torch.Tensor(noise), requires_grad=False)

        return _noise_tensor
    
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        state = self._encode(document)
        
        state["source_tokens"] = document["tokens"]
        state["target_tokens"] = query["tokens"]
                
        state = self._init_decoder_state(state)
        logprobs = self._forward_loop(state, query)

        return logprobs

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder.forward(embedded_input, source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        
        encoder_outputs = state["encoder_outputs"]
        
        if self.enc_lstm_num_layers > 0:
            final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                                 state["source_mask"],
                                                                 self._encoder.is_bidirectional())
            # Initialize the decoder hidden state with the final output of the encoder.
            # shape: (batch_size, decoder_output_dim)
            state["decoder_hidden"] = final_encoder_output
        else:
            state["decoder_hidden"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
            
        state["decoder_context"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        num_decoding_steps = self._max_decoding_steps
        
        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"] # <-- IN for our TARGET EMBEDDER ##
        _, target_sequence_length = targets.size()

        num_decoding_steps = target_sequence_length - 1

        decoder_final_list: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            input_choices = targets[:, timestep]
            step_decoder_final, state = self._prepare_output_projections(input_choices, state)
            decoder_final_list.append(step_decoder_final)
            
        decoder_finals = decoder_final_list[0].new_zeros((decoder_final_list[0].size(0), decoder_final_list[0].size(1),
                                                          len(decoder_final_list)))
        for i, p in enumerate(decoder_final_list):
            decoder_finals[:, :, i] = p
        _decoder_finals = decoder_finals.transpose(1, 2)
        _target = state["target_tokens"][:, 1:]
        _target = _target.contiguous()

        #logprobs = self.get_logprobs(logproba, state["target_tokens"])
        logprobs = self._output_projection_layer(input=_decoder_finals, target=_target)
        
        return logprobs
    
    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_outputs = state["encoder_outputs"]
        source_mask = state["source_mask"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]

        decoder_embedded_input = self._target_embedder.forward({"tokens":last_predictions})
        decoder_hidden, decoder_context = self._decoder_cell(decoder_embedded_input, (decoder_hidden, decoder_context))

        attn_scores = self._attention.forward(decoder_hidden, encoder_outputs, source_mask)
        attn_context = util.weighted_sum(encoder_outputs, attn_scores)
        
        decoder_output = torch.cat((attn_context, decoder_hidden), -1)
        decoder_final = self._hidden_projection_layer(decoder_output)
        
        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_scores"] = attn_scores
        state["attn_context"] = attn_context

        return decoder_final, state

    
'''        



