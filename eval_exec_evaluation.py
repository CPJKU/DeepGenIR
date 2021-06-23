import time
import logging
import os
import pdb
import sys
from datetime import datetime
import numpy as np
import pickle
import argparse, os, shutil, hashlib
from datetime import datetime

from eval_scripts import *
from eval_evaluation_tool import EvaluationToolTrec

parser = argparse.ArgumentParser()
parser.add_argument('--trecevalpath', type=str, default="/share/rk0/home/navid/trec_eval/trec_eval")
parser.add_argument('--qrels_path', type=str, required=True)
parser.add_argument('--run_file_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--save_files_prefix', type=str, default="", 
                    help="Prefix to be added at the beginning of .txt and .pkl files")
parser.add_argument('--trec_measures_param', default="-m ndcg -m ndcg_cut -m recall -m P -m map -m recip_rank", type=str)
parser.add_argument('--test', action='store_true',
                    help='Evaluate test set')

args = parser.parse_args()

## LOGGER
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


evaluator = EvaluationToolTrec(trec_eval_path=args.trecevalpath, qrel_path=args.qrels_path,
                               trec_measures_param=args.trec_measures_param)
    
logger.info ("Evaluating: %s" % args.run_file_path)

result_info = compute_metrics_from_files(evaluator, args.run_file_path)
logger.info ("Results: %s" % str(result_info["metrics_avg"]))

if args.test:
    file_save_path = args.save_files_prefix + 'test-metrics.txt'
    file_save_pkl_path = args.save_files_prefix + 'test-metrics.pkl'
else:
    file_save_path = args.save_files_prefix + 'best-validation-metrics.txt'
    file_save_pkl_path = args.save_files_prefix + 'best-validation-metrics.pkl'

result_info_tosave = {"cs@n": result_info["cs@n"],
                      "metrics_avg": result_info["metrics_avg"],
                      "metrics_perq": result_info["metrics_perq"]}
    
with open(os.path.join(args.save_dir, file_save_path), "w") as fw:
    fw.write("{'cs@n':%d, 'metrics_avg':%s}" % (result_info_tosave["cs@n"], result_info_tosave["metrics_avg"]))
with open(os.path.join(args.save_dir, file_save_pkl_path), "wb") as fw:
    pickle.dump(result_info_tosave, fw)
    

