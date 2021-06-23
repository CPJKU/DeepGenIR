# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict, List
from typing import Callable
import logging
import pdb
import numpy as np

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, LabelField, ArrayField
#from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.instance import Instance

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

                                            
@DatasetReader.register("ir_triple_transformers_loader")
class IrTripleTransformersDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    `bos_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_bos_token : bool, (optional, default=True)
        Whether or not to add `bos_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 transformers_tokenizer: PreTrainedTokenizer,
                 add_special_tokens: bool = False,
                 max_doc_length: int = -1, # in words
                 max_query_length:int = -1,
                 lazy: bool = False,
                 preprocess: Callable = None,
                 add_bos_token_to_decoder: bool = True,
                 bos_token_id: int = -1,
                 max_out_length_in_tokens: int = None) -> None:
        super().__init__(lazy)
        #self._pre_tokenizer = WhitespaceTokenizer()
        self._transformers_tokenizer = transformers_tokenizer
        self._add_special_tokens = add_special_tokens
        self._max_doc_length = max_doc_length
        self._max_query_length = max_query_length
        self._preprocess = preprocess
        self.add_bos_token_to_decoder = add_bos_token_to_decoder
        self.bos_token_id = bos_token_id
        self._max_out_length = max_out_length_in_tokens

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue
 
                line_parts = line.split('\t')
                if len(line_parts) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                if self._preprocess != None:
                    query_sequence = self._preprocess(query_sequence)
                    doc_pos_sequence = self._preprocess(doc_pos_sequence)
                    doc_neg_sequence = self._preprocess(doc_neg_sequence)
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore

        _truncate_encoder = False
        _truncate_encoder_max_length = self._max_out_length
        _truncate_decoder = False
        _truncate_decoder_max_length = self._max_out_length
        if _truncate_decoder_max_length is not None:
            _truncate_decoder = True
            if self.add_bos_token_to_decoder:
                _truncate_decoder_max_length -= 1 # take into account <bos> token to be added in the end

        if len(query_sequence.strip()) == 0:
            query_sequence = "@@UNKNOWN@@"
        
        query_pre_tokenized = query_sequence.split()

        if self._max_query_length > -1: # in words
            query_pre_tokenized = query_pre_tokenized[:self._max_query_length]

        query_tokenized = self._transformers_tokenizer(' '.join(query_pre_tokenized),
                                                       truncation = _truncate_decoder,
                                                       add_special_tokens = self._add_special_tokens,
                                                       max_length = _truncate_decoder_max_length)["input_ids"]
        
        doc_pos_pre_tokenized = doc_pos_sequence.split()
        
        if self._max_doc_length > -1: # in words
            doc_pos_pre_tokenized = doc_pos_pre_tokenized[:self._max_doc_length]
            
        doc_pos_tokenized = self._transformers_tokenizer(' '.join(doc_pos_pre_tokenized),
                                                         truncation = _truncate_encoder,
                                                         add_special_tokens = self._add_special_tokens,
                                                         max_length = _truncate_encoder_max_length)["input_ids"]
        
        doc_neg_pre_tokenized = doc_neg_sequence.split()
        
        if self._max_doc_length > -1: # in words
            doc_neg_pre_tokenized = doc_neg_pre_tokenized[:self._max_doc_length]    
            
        doc_neg_tokenized = self._transformers_tokenizer(' '.join(doc_neg_pre_tokenized),
                                                         truncation = _truncate_encoder, 
                                                         add_special_tokens = self._add_special_tokens,
                                                         max_length = _truncate_encoder_max_length)["input_ids"]

        # ADAPT ADDITNG bos AND END
        if self.add_bos_token_to_decoder:
            query_tokenized.insert(0, self.bos_token_id)
            #query_tokenized.append(self.eos_token_id)

            #doc_pos_tokenized.insert(0, self.bos_token_id)
            #doc_pos_tokenized.append(self.eos_token_id)

            #doc_neg_tokenized.insert(0, self.bos_token_id)
            #doc_neg_tokenized.append(self.eos_token_id)

        query_field = ArrayField(np.array(query_tokenized))
        doc_pos_field = ArrayField(np.array(doc_pos_tokenized))
        doc_neg_field = ArrayField(np.array(doc_neg_tokenized))

        return Instance({
            "query_tokens" : query_field,
            "doc_pos_tokens" : doc_pos_field,
            "doc_neg_tokens" : doc_neg_field })
