import numpy as np
from typing import Dict, List, Tuple, Optional, overload


import torch
from torch import Tensor

from torch.nn.modules.linear import Linear

from allennlp.models import Model
from allennlp.nn import util

from transformers import BertModel




class DiscriminativeBert (Model):
    def __init__(self,
                 bert: BertModel,):
        super(DiscriminativeBert, self).__init__(vocab=None) # (?)
        self._bert = bert
        self._embedding_size = self._bert.config.hidden_size 
        self._output_projection_layer = Linear(self._embedding_size, 2)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        max_input_length = 512

        bsz = document["tokens"].size(0)
        # doc_max_length = document["tokens"].size(1)
        # qry_max_length = query["tokens"].size(1)
        
        tok_seq = torch.full((bsz, max_input_length), 0, dtype=int).cuda()
        seg_mask = torch.full((bsz, max_input_length), 0, dtype=int).cuda()
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
            ## we assume that length of query (+2) never exceeds <max_input_length>
            ## therefore we only truncate the document
            ## in extreme cases this can hurt
            _vec = document["tokens"][batch_i]
            _length = len(_vec[_vec != 0])
            _fill_until = _length + _offset
            if _fill_until >= max_input_length:
                _fill_until = max_input_length - 1 # leaving space for the last <sep> 
                _length = _fill_until - _offset
            tok_seq[batch_i, _offset:_fill_until] = document["tokens"][batch_i, :_length]
            seg_mask[batch_i, _offset:_fill_until] = torch.full((_length, 1), seg_2_value, dtype=int)[:, 0]
            _offset += _length
            
            tok_seq[batch_i, _offset:_offset+1] = SEP_id
            seg_mask[batch_i, _offset:_offset+1] = seg_2_value
            _offset += 1
            
        #cls_vec = torch.full((document["tokens"].shape[0],1), 101, dtype=int).cuda()
        #sep_vec = torch.full((document["tokens"].shape[0],1), 102, dtype=int).cuda()

        #tok_seq1 = torch.cat((cls_vec,query["tokens"], sep_vec), 1)
        #tok_seq2 = torch.cat((document["tokens"], sep_vec), 1)
        #seg1 = torch.full((tok_seq1.shape), 0, dtype=int)
        #seg2 = torch.full((tok_seq2.shape), 1, dtype=int)

        #tok_seq = torch.cat((tok_seq1, tok_seq2), 1).cuda()
        #seg_mask = torch.cat((seg1,seg2),1).cuda()
        
        pad_mask = util.get_text_field_mask({"tokens":tok_seq}).cuda()

        out = self._bert(input_ids=tok_seq, attention_mask=pad_mask, token_type_ids=seg_mask)
        scores = self._output_projection_layer(out[0][:,0,:])

        lprobs = torch.nn.LogSoftmax(dim=-1)(scores)
        return {"rels" : lprobs[:,0], "logprobs": lprobs}

        
