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

from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model
from allennlp.nn import util

from transformers import BartModel

from models.positional_encoding import PositionalEncoding

class Bart(Model):
    def __init__(self,
                 bartmodel: BartModel) -> None:
        super(Bart, self).__init__(vocab=None) # (?)

        self.bart = bartmodel
        self.vocab_size = self.bart.config.vocab_size
        self.num_classes = self.vocab_size
        
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        _input_ids = document["tokens"]
        _decoder_input_ids = query["tokens"]
                
        _labels = _decoder_input_ids.clone()
        _labels[:, :-1] = _decoder_input_ids[:, 1:]
        _labels[:, -1] = 0
        
         # 1 - not masked, 0 - masked;
        _input_padding_mask = util.get_text_field_mask(document)
        _decoder_padding_mask = util.get_text_field_mask(query)

        output_dict = self.bart(input_ids=_input_ids, attention_mask=_input_padding_mask, decoder_input_ids=_decoder_input_ids, decoder_attention_mask=_decoder_padding_mask, labels=_labels, return_dict=True)
        
        _logproba = torch.nn.LogSoftmax(dim=-1)(output_dict.logits)
        _logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba.transpose(1, 2), _labels)
        _logprobs_batch = torch.sum(_logprobs, dim=1)
        
        return {"rels":_logprobs_batch}

    