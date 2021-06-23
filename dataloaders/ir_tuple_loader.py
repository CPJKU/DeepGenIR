# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict, List
from typing import Callable
import logging
import sys
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ir_tuple_loader")
class IrTupleDatasetReader(DatasetReader):
    """
    Read a tsv file containing tuples of query and document sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_id>\t<doc_id>\t<query_sequence_string>\t<doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_id: ``LabelField`` and
        doc_id: ``LabelField`` and
        query_tokens: ``TextField`` and
        doc_tokens: ``TextField`` and
        query_length: ``LabelField`` and
        doc_length: ``LabelField``
        
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
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 lazy: bool = False,
                 preprocess: Callable = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
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
        try:
            with open(cached_path(file_path), "r", encoding="utf8") as data_file:
                #logger.info("Reading instances from lines in file at: %s" % file_path)
                for line_num, line in enumerate(data_file):
                    line = line.strip("\n")

                    if not line:
                        continue

                    line_parts = line.split('\t')
                    if len(line_parts) != 4:
                        sys.stdout.write ("Invalid line format: %s (line number %d)\n" % (line, line_num + 1))
                        sys.stdout.flush()
                        raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                    query_id, doc_id, query_sequence, doc_sequence = line_parts
                    if self.preprocess != None:
                        query_sequence = self.preprocess(query_sequence)
                        doc_sequence = self.preprocess(doc_sequence)
                    _instance = self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence)
                    
                    yield _instance
        except Exception as e: 
            sys.stdout.write(e)
            sys.stdout.flush()

    @overrides
    def text_to_instance(self, query_id:str, doc_id:str, query_sequence: str, doc_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = MetadataField(int(query_id))
        doc_id_field = MetadataField(doc_id)
        
        # dummy code to prevent empty queries
        if len(query_sequence.strip()) == 0:
            query_sequence = "@@UNKNOWN@@"

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        #if self._source_add_start_token:
        query_tokenized.insert(0, Token(START_SYMBOL))
        query_tokenized.append(Token(END_SYMBOL))
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]
        
            

        query_field = TextField(query_tokenized, self._token_indexers)
        
        doc_tokenized = self._tokenizer.tokenize(doc_sequence)
        doc_tokenized.insert(0, Token(START_SYMBOL))
        doc_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_tokenized = doc_tokenized[:self.max_doc_length]

        doc_field = TextField(doc_tokenized, self._token_indexers)

        query_length = LabelField(len(query_tokenized), skip_indexing=True)
        doc_length = LabelField(len(doc_tokenized), skip_indexing=True)

        # generating local-indexed versions of query and positive document same as in ir_triple_loader.py:
        # query and doc are concatenated, indexed together locally (anew, starting from 0)
        # and then returned separately
        # required for Pointer Generator Network in order to tell difference between unknown words;
        query_and_doc_in_local_dict = self._tokens_to_ids(query_tokenized + doc_tokenized)
        # indexing is done with lower-case applied
        query_in_local_dict = query_and_doc_in_local_dict[:len(query_tokenized)]
        doc_in_local_dict = query_and_doc_in_local_dict[len(query_tokenized):]

        query_local_field = ArrayField(np.array(query_in_local_dict))
        doc_local_field = ArrayField(np.array(doc_in_local_dict))
        return Instance({
            "query_id":query_id_field,
            "doc_id":doc_id_field,

            "query_tokens":query_field,
            "query_local_dict" : query_local_field,

            "doc_tokens":doc_field,
            "doc_local_dict" : doc_local_field,
            
            "query_length":query_length,
            "doc_length":doc_length})


