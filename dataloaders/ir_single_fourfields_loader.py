# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py
from typing import Dict
from typing import Callable
import logging
import pdb

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MAX_TEXT_LENGTH = 100000

@DatasetReader.register("irSingle4Fields")
class IrSingleFourfieldsDatasetReader(DatasetReader):
    """
    Read a tsv file containing single data sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <id>\t<url>\t<title>\t<text>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        id: ``LabelField`` and
        url: ``TextField`` and
        title_tokens: ``TextField``
        text_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the sequence.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 lowercase: bool = True,
                 lazy: bool = False,
                 preprocess: Callable = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() #word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=lowercase)}
        self._add_start_token = add_start_token
        self.preprocess = preprocess

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 4:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                sequenceid, url, title, text = line_parts
                if self.preprocess != None:
                    title = self.preprocess(title)
                    text = self.preprocess(text)
                yield self.text_to_instance(sequenceid, url, title, text)

    @overrides
    def text_to_instance(self, sequenceid: str, url: str, title: str, text: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        
        id_field = MetadataField(sequenceid)
        
        url_field = MetadataField(url)
        
        tokenized_title = self._tokenizer.tokenize(title)
        title_field = TextField(tokenized_title, self._token_indexers)

        if len(text) > MAX_TEXT_LENGTH:
            print ("ID %s is longer than %d characters. The string is trimed to max length" % (sequenceid, MAX_TEXT_LENGTH))
        tokenized_text = self._tokenizer.tokenize(text[:MAX_TEXT_LENGTH])
        text_field = TextField(tokenized_text, self._token_indexers)

        
        return Instance({"id": id_field, "url": url_field, "title_tokens": title_field, "text_tokens": text_field})
    