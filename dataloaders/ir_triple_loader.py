# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict, List
from typing import Callable
import logging
import pdb
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
                                            
@DatasetReader.register("ir_triple_loader")
class IrTripleDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 query_add_sep_token: bool = False, 
                 document_add_sep_token: bool = False,
                 query_add_cls_token: bool = False,
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 lazy: bool = False,
                 preprocess: Callable = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.query_add_sep_token = query_add_sep_token
        self.document_add_sep_token = document_add_sep_token
        self.query_add_cls_token = query_add_cls_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.preprocess = preprocess

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids = dict()
        out = list()
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

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
                if self.preprocess != None:
                    query_sequence = self.preprocess(query_sequence)
                    doc_pos_sequence = self.preprocess(doc_pos_sequence)
                    doc_neg_sequence = self.preprocess(doc_neg_sequence)
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        query_tokenized = self._tokenizer.tokenize(query_sequence)
        #if self._source_add_start_token:
        query_tokenized.insert(0, Token(START_SYMBOL))
        query_tokenized.append(Token(END_SYMBOL))
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]
        if self.query_add_sep_token:
            query_tokenized.insert(0, Token(SEP_SYMBOL))
        if self.query_add_cls_token:
            query_tokenized.insert(0, Token(CLS_SYMBOL))
        
        query_field = TextField(query_tokenized, self._token_indexers)

        doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
        doc_pos_tokenized.insert(0, Token(START_SYMBOL))
        doc_pos_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_pos_tokenized = doc_pos_tokenized[:self.max_doc_length]
        if self.document_add_sep_token:
            doc_pos_tokenized.insert(0, Token(SEP_SYMBOL))
        doc_pos_field = TextField(doc_pos_tokenized, self._token_indexers)
        
        doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)
        doc_neg_tokenized.insert(0, Token(START_SYMBOL))
        doc_neg_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_neg_tokenized = doc_neg_tokenized[:self.max_doc_length]
        if self.document_add_sep_token:
            doc_neg_tokenized.insert(0, Token(SEP_SYMBOL))
        doc_neg_field = TextField(doc_neg_tokenized, self._token_indexers)

        query_length = LabelField(len(query_tokenized), skip_indexing=True)
        doc_pos_length = LabelField(len(doc_pos_tokenized), skip_indexing=True)
        doc_neg_length = LabelField(len(doc_neg_tokenized), skip_indexing=True)
        
        # generating local-indexed versions of query and positive document:
        # query and doc are concatenated, indexed together locally (anew, starting from 0)
        # and then returned separately
        # required for Pointer Generator Network in order to tell difference between unknown words;
        query_and_docs_in_local_dict = self._tokens_to_ids(query_tokenized + doc_pos_tokenized + doc_neg_tokenized)
        # indexing is done with lower-case applied
        query_in_local_dict = query_and_docs_in_local_dict[:len(query_tokenized)]
        doc_pos_in_local_dict = query_and_docs_in_local_dict[len(query_tokenized):len(query_tokenized)+len(doc_pos_tokenized)]
        doc_neg_in_local_dict = query_and_docs_in_local_dict[len(query_tokenized)+len(doc_pos_tokenized):]

        query_local_field = ArrayField(np.array(query_in_local_dict))
        doc_pos_local_field = ArrayField(np.array(doc_pos_in_local_dict))
        doc_neg_local_field = ArrayField(np.array(doc_neg_in_local_dict))

        #print(query_tokenized, "\n", query_in_local_dict, len(query_tokenized), "vs", len(query_in_local_dict) ,"\n  ===")
        #print(doc_pos_tokenized, "\n", doc_pos_in_local_dict, len(doc_pos_tokenized), "vs", len(doc_pos_in_local_dict) ,"\n  ===")
        #print(doc_neg_tokenized, "\n", doc_neg_in_local_dict, len(doc_neg_tokenized), "vs", len(doc_neg_in_local_dict) ,"\n  ===")

        return Instance({
            "query_tokens" : query_field,
            "query_local_dict" : query_local_field,
            "doc_pos_tokens" : doc_pos_field,
            "doc_pos_local_dict" : doc_pos_local_field,
            "doc_neg_tokens" : doc_neg_field,
            "doc_neg_local_dict" : doc_neg_local_field,
            "query_length" : query_length,
            "doc_pos_length" : doc_pos_length,
            "doc_neg_length" : doc_neg_length})
