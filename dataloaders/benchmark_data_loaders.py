#
# benchmark setup to measer time per component in the batch generation 
# -------------------------------
#
# usage:

import argparse
import os
import sys
sys.path.append(os.getcwd())

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from utils import *
from dataloaders.ir_triple_loader import *
#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--dataset-file', action='store', dest='dataset_file',
                    help='dataset file: for triple loader', required=True)
parser.add_argument('--vocab-file', action='store', dest='vocab_file',
                    help='vocab directory path', required=True)

args = parser.parse_args()


#
# load data & create vocab
# -------------------------------
#  

loader = IrTripleDatasetReader(lazy=True,tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()))

instances = loader.read(args.dataset_file)
_iterator = BucketIterator(batch_size=64,
                           sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])
_iterator.index_with(Vocabulary.from_files(args.vocab_file))

with Timer("iterate over all"):
    for i in _iterator(instances, num_epochs=1):
        test = 0