import yaml
from timeit import default_timer
from typing import Dict
from datetime import datetime
import os
import logging
import argparse
import shutil
import pickle
import pdb

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Timer():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = default_timer()
        print(self.message + " started ...")

    def __exit__(self, type, value, traceback):
        print(self.message+" finished, after (s): ",
              (default_timer() - self.start_time))

def get_config(config_path:str, overwrites:str = None) ->Dict[str, any] :
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if overwrites is not None and overwrites != "":
        over_parts = [yaml.load(x, Loader=yaml.FullLoader) for x in overwrites.split(",")]
        
        for d in over_parts:
            for key, value in d.items():
                cfg[key] = value

    return cfg

def save_config(config_path:str, config:Dict[str, any]):
    with open(config_path, 'w') as ymlfile:
        yaml.safe_dump(config, ymlfile,default_flow_style=False)

def load_config(config_path:str) ->Dict[str, any] :
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', action='store', dest='run_name',
                        help='run name, used for the run folder (no spaces, special characters)', required=False)
    parser.add_argument('--run-folder', action='store', dest='run_folder',
                        help='run folder if it exists, if not set a new one is created using run-name', required=False)
    parser.add_argument('--continue_train', action='store_true',
                        help='continue train from a checkpoint')
    parser.add_argument('--test', action='store_true',
                        help='only runs the code for testing pre-trained models; run-folder should be set ')
    
    parser.add_argument('--config-file', action='store', dest='config_file',
                        help='config file with all hyper-params & paths', required=False)
    parser.add_argument('--config-overwrites', action='store', dest='config_overwrites',
                        help='overwrite config values -> key1: valueA,key2: valueB ', required=False)

    parser.add_argument('--gpu-id', action='store', dest='cuda_device_id', type=int, default=0,
                    help='optional cuda device id for multi gpu parallel runs of train.py', required=False)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    #
    parser.add_argument('--test-files-pretfix', action='store', dest='test_files_prefix',
                        help='prefix to be added at the start of test file names')
    #
    parser.add_argument('--custom-test-depth', action='store', dest='custom_test_depth',
                        help='set custom depth reranking for testing')                    
    parser.add_argument('--custom-test-tsv', action='store', dest='custom_test_tsv',
                        help='sets new test path, overrides config settings')
    parser.add_argument('--custom-test-qrels', action='store', dest='custom_test_qrels',
                        help='sets new path for qrels, overrides config settings')
    parser.add_argument('--custom-test-candidates', action='store', dest='custom_test_candidates',
                        help='sets new path for BM25 candidates, overrides config settings')

    
    return parser

def get_logger_to_file(run_folder,name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger.setLevel(logging.INFO)

    log_filepath = os.path.join(run_folder, 'log.txt')
    file_hdlr = logging.FileHandler(log_filepath)
    file_hdlr.setFormatter(formatter)
    file_hdlr.setLevel(logging.INFO)
    logger.addHandler(file_hdlr)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def prepare_experiment_folder(base_path, run_name):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')[:-4]

    run_folder = os.path.join(base_path,time_stamp + "_" + run_name)

    os.makedirs(run_folder)

    return run_folder

def prepare_experiment(args):
    if (args.continue_train or args.test) and (args.run_folder is None):
        raise Exception("'continue_train' or 'test' are set. 'run_folder' also need to be set")
    
    if (args.continue_train or args.test):
        run_folder = args.run_folder
        config = load_config(os.path.join(run_folder, "config.yaml"))
    else:
        config = get_config(os.path.join(os.getcwd(), args.config_file), args.config_overwrites)
        
        if args.run_folder is not None:
            run_folder = args.run_folder
        else:
            if args.debug:
                _base_path = config["debug_base_path"]
            else:
                _base_path = config["expirement_base_path"]
                
            if args.run_name is None:
                args.run_name = '%s#%s#v%s' % (config["model"], config["token_embedder_type"], 
                                                   config["vocab_directory"].split("_")[-1])

            run_folder = prepare_experiment_folder(_base_path, args.run_name)

        save_config(os.path.join(run_folder, "config.yaml"), config)
        
        # copy source code of matchmaker
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #shutil.copytree(dir_path, os.path.join(run_folder,"matchmaker-src"), ignore=shutil.ignore_patterns("__pycache__"))

        if args.debug:
            config["log_interval"] = 10
            config["eval_log_interval"] = 50
            config["max_training_batch_count"] = 30
            config["max_validation_batch_count"] = 500
            config["validate_every_n_batches"] = 10
            config["max_test_batch_count"] = 100
            config["epochs"] = 2
            
    return run_folder, config

def parse_reference_set(file_path, to_N):
    reference_set_rank = {} # dict[qid][did] -> rank
    reference_set_tuple = {} # dict[qid] -> sorted list of (did, score)

    
    with open(file_path, "r") as cs_file:
        for line in cs_file:
            vals = line.rstrip().split(' ') # 8 Q0 8383396 1 16.144300 Anserini
            
            rank = int(vals[3])
            
            if rank <= to_N:
                
                #q_id = int(vals[0])
                q_id = vals[0]
                d_id = vals[2]
                score = float(vals[4])

                if q_id not in reference_set_rank:
                    reference_set_rank[q_id] = {}
                if q_id not in reference_set_tuple:
                    reference_set_tuple[q_id] = []

                reference_set_rank[q_id][d_id] = rank
                reference_set_tuple[q_id].append((d_id, score))

    return reference_set_rank, reference_set_tuple

def checkpoint_save(filepath, model, criterion, optimizer, epoch, batch):
    with open(filepath, 'wb') as f:
        torch.save([model.state_dict(), criterion.state_dict(), optimizer.state_dict(), epoch, batch], f)

def checkpoint_load(filepath):
    with open(filepath, 'rb') as f:
        model_state, criterion_state, optimizer_state, epoch, batch = torch.load(f)
    return model_state, criterion_state, optimizer_state, epoch, batch

def model_save(filepath, model, best_result_info):
    with open(filepath, 'wb') as f:
        torch.save([model.state_dict(), best_result_info], f)

def model_load(filepath, _GPU_n = None):
    with open(filepath, 'rb') as f:
        if _GPU_n != None:
            model_state, best_result_info = torch.load(f,map_location=lambda storage, loc: storage.cuda(_GPU_n))
        else:
            model_state, best_result_info = torch.load(f)
    return model_state, best_result_info

def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

def get_idf_lookup(idfcf_path, vocab):
    ## loading IDF values
    with open (idfcf_path, 'rb') as fr:
        idfcf_dic = pickle.load(fr)

    _idfs_unnorm = []
    for v_i in range(vocab.get_vocab_size()):
        v = vocab.get_token_from_index(v_i)
        if v not in idfcf_dic:
            continue
        _idfs_unnorm.append(idfcf_dic[v][0])

    _idfs_unnorm = np.array(_idfs_unnorm).reshape(-1, 1)
    _scaler = MinMaxScaler()
    _scaler.fit(_idfs_unnorm)
    _idfs = _scaler.transform(_idfs_unnorm).squeeze(-1)

    _idfs_vocab = []
    _idfs_vocab_cnt = 0
    for v_i in range(vocab.get_vocab_size()):
        v = vocab.get_token_from_index(v_i)
        if v in idfcf_dic:
            _idfs_vocab.append(_idfs[_idfs_vocab_cnt])
            _idfs_vocab_cnt += 1
        else:
            if (v == '@@UNKNOWN@@'): #padding and unknown
                _idfs_vocab.append(0.9) # a high value
            elif (v == '@@PADDING@@'): #padding and unknown
                _idfs_vocab.append(0.01) # some low value
            else: # other words
                _idfs_vocab.append(0.01) # some low value
    idf_lookup = torch.nn.Embedding(vocab.get_vocab_size(), 1)
    idf_lookup.weight = nn.Parameter(torch.Tensor(_idfs_vocab), requires_grad=False)

    return idf_lookup

#
# from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# Thanks!
#
class EarlyStopping():
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        self.stop = False
        #if patience == 0:
        #    self.is_better = lambda a, b: True
        #    self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            self.stop = True
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.stop = True
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100) 
                
                
