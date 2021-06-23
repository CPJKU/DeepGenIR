import os
import copy
import time
import glob
from typing import Dict, Tuple, List
import pdb
import numpy as np
import pickle
import torch

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from eval_scripts import *
from multiprocess_input_pipeline import *

#
# raw model evaluation, returns model results as python dict, does not save anything / no metrics
#
def evaluate_model(model, cuda_device, eval_tsv, max_eval_batch_count, config, logger, vocabulary):

    model.eval()  # turning off training
    eval_results = {}
    eval_processes = []
    uncertainty_scores = {}

    try:
        eval_files = glob.glob(eval_tsv)
        eval_queue, eval_processes, eval_exit = get_multiprocess_batch_queue("eval-batches",
                                                                             multiprocess_validation_loader,
                                                                             files=eval_files,
                                                                             conf=config,
                                                                             _logger=logger,
                                                                             queue_size=200,
                                                                             vocabulary=vocabulary)
        #time.sleep(len(eval_processes))  # fill the queue
        start_time = time.time()
        eval_log_interval = config["eval_log_interval"]
        batch_num = 0
        batch_null_cnt = 0

        total_inference_time = 0.0
        total_decoding_time = 0.0
        total_encoding_time = 0.0
        total_softmax_time = 0.0
        
        uncertainty_scores = {} # list to keep track of uncertainty and rels

        with torch.no_grad():
            while (True):
                batch_orig = eval_queue.get()
                if batch_orig is None:
                    batch_null_cnt += 1
                    if batch_null_cnt == len(eval_files):
                        break
                    else:
                        continue
                if batch_num >= max_eval_batch_count and max_eval_batch_count != -1:
                    break

                if config["token_embedder_type"] in ["bert", "bart"]:
                    # adapting ir_triple_transformers_loader format to allennlp format used in standard loaders;
                    batch_orig = {"query_id" : batch_orig["query_id"],
                                  "doc_id" : batch_orig["doc_id"],
                                  "query_tokens" : {"tokens" : batch_orig["query_tokens"].long()},
                                  "doc_tokens" : {"tokens" : batch_orig["doc_tokens"].long()}}
                
                batch = copy.deepcopy(batch_orig)
                if cuda_device != -1:
                    batch = move_to_device(batch, cuda_device)
                
                forward_in_time = time.time()
                if config["model"].startswith("pgn"):
                    output_dict = model.forward(target = batch["query_tokens"],
                                           target_local_token_ids = batch["query_local_dict"],
                                           source = batch["doc_tokens"],
                                           source_local_token_ids = batch["doc_local_dict"])
                                    ### local ditionaries allow for differentiation between multiple unknown words in the input
                                    # (words not from our vocabulary)
                else:
                    output_dict = model.forward(batch["query_tokens"], batch["doc_tokens"])
                forward_time = time.time() - forward_in_time

                total_inference_time += forward_time
                if "dec_time" in output_dict:
                    total_decoding_time += output_dict["dec_time"]
                if "enc_time" in output_dict:
                    total_encoding_time += output_dict["enc_time"]
                if "softmax_time" in output_dict:
                    total_softmax_time += output_dict["softmax_time"]

                
                output = output_dict["rels"]
                output = output.cpu()  # get the output back to the cpu - in one piece

                for sample_i, sample_query_id in enumerate(batch_orig["query_id"]):  # operate on cpu memory
                    sample_query_id = sample_query_id
                    sample_doc_id = batch_orig["doc_id"][sample_i]  # again operate on cpu memory

                    if sample_query_id not in eval_results:
                        eval_results[sample_query_id] = {}
                        if "uncertainty" in output_dict:
                            uncertainty_scores[sample_query_id] = {}

                    # pdb.set_trace()         
                    eval_results[sample_query_id][sample_doc_id] = float(output[sample_i])
                    if "uncertainty" in output_dict:
                        uncertainty_scores[sample_query_id][sample_doc_id] = {}
                        uncertainty_scores[sample_query_id][sample_doc_id]["rel"] = float(output[sample_i])
                        uncertainty_scores[sample_query_id][sample_doc_id]["uncertainty"] = output_dict["uncertainty"][sample_i].tolist()
                if batch_num % eval_log_interval == 0:
                    #logging
                    logger.info('EVALUATION | %5d batches' % (batch_num))
                    start_time = time.time()
                    if eval_queue.qsize() < 10:
                        logger.warning("evaluation_queue.qsize() < 10 (%d)" % eval_queue.qsize())

                batch_num += 1

        logger.info('EVALUATION FINISHED | %5d batches ' % (batch_num))
        logger.info("Total inference time: %f" % (total_inference_time))
        logger.info("Total decoding  time: %f" % (total_decoding_time))
        logger.info("Total encoding  time: %f" % (total_encoding_time))
        logger.info("Total softmax   time: %f" % (total_softmax_time))


        # make sure we didn't make a mistake in the configuration / data preparation
        if eval_queue.qsize() != 0 and max_eval_batch_count == -1:
            logger.error("evaluation_queue.qsize() is not empty (%d) after evaluation" % eval_queue.qsize())

        eval_exit.set()  # allow sub-processes to exit

        for proc in eval_processes:
            if proc.is_alive():
                proc.terminate()

    except BaseException as e:
        logger.exception('[eval_model] Got exception: %s' % str(e))

        for proc in eval_processes:
            if proc.is_alive():
                proc.terminate()
        raise e
    #pdb.set_trace()
    return {"scores": eval_results, "uncertainty": uncertainty_scores}

#
# validate a model during training + save results, metrics 
#
def validate_model(model, config, logger, run_folder, cuda_device, writer, epoch_number, batch_cnt_global, metric_tocompare,
                   evaluator, reference_set_rank, reference_set_tuple, vocabulary, batch_number=-1, global_best_info=None):

    
    # this means we currently after a completed batch
    if batch_number == -1:
        evaluation_id = str(epoch_number)

    # this means we currently are in an inter-batch evaluation
    else:
        evaluation_id = str(epoch_number) + "-" +str(batch_number)

    logger.info("[VALIDATION Epoch-Batch %s] --- Start" % evaluation_id)

    eval_res = evaluate_model(model, cuda_device, config["validation_tsv"], 
                                            config["max_validation_batch_count"], config, logger, vocabulary)
    validation_rank_scores = eval_res["scores"]


    #
    # save sorted results
    #
    validation_file_path = os.path.join(run_folder, "validation-run-"+evaluation_id+"-full-rerank.txt")
    save_sorted_results(validation_rank_scores, validation_file_path)

    #
    # compute ir metrics (for ms_marco) and output them (to the logger + own csv file)
    # ---------------------------------
    #
    best_result_info = {}
    best_result_info["epoch"] = epoch_number
    best_result_info["batch_number"] = batch_number

    
    #
    # do a cs@n over multiple n evaluation
    #
    if reference_set_rank != None:
        logger.info("Validation Epoch-Batch %s: computing metrics in range %s inervals %d" % 
                    (evaluation_id, str(config["validation_candidate_set_from_to"]),
                     config["validation_candidate_set_interval"]))

        _best_result_info, _metric_results_avg = compute_metrics_threshold_range(evaluator, validation_file_path,
                                                                                 reference_set_rank, reference_set_tuple,
                                                                                 config["validation_candidate_set_from_to"],
                                                                                 config["validation_candidate_set_interval"],
                                                                                 metric_tocompare=metric_tocompare)
        best_result_info["metrics_avg"] = _best_result_info["metrics_avg"]
        best_result_info["metrics_perq"] = _best_result_info["metrics_perq"]
        best_result_info["rank_score"] = _best_result_info["rank_score"]
        best_result_info["cs@n"] = _best_result_info["cs@n"]

        # save evaluation metric overview
        for cs_at_n, v in _metric_results_avg.items():
            writer.add_scalar("%s validation/cs_at_n" % metric_tocompare, v[metric_tocompare], cs_at_n)

    #
    # do a 1x evaluation over the full given validation set
    #
    else:
        _best_result_info = compute_metrics_from_files(validation_file_path)
        best_result_info["metrics_avg"] = _best_result_info["metrics_avg"]
        best_result_info["metrics_perq"] = _best_result_info["metrics_perq"]
        best_result_info["rank_score"] = _best_result_info["rank_score"]
        best_result_info["cs@n"] = _best_result_info["cs@n"]

    #
    # remove validation results file
    #
    if config["validation_save_only_best"] == True and global_best_info != None:
        os.remove(validation_file_path)

    logger.info("Validation Epoch-Batch %s %s at rank %d: %f" % (evaluation_id, metric_tocompare, best_result_info["cs@n"],
                                                                 best_result_info["metrics_avg"][metric_tocompare]))

    return best_result_info

#
# test a model after training + save results, metrics 
#
def test_model(args, model, config, logger, run_folder, cuda_device, metric_tocompare, evaluator,
               reference_set_rank, reference_set_tuple, vocabulary, reference_set_n=None):

    logger.info("[TEST] --- Start")

    if args.custom_test_tsv:
        logger.info("OVERRIDING: using custom test_tsv: %s" % (args.custom_test_tsv))
        _test_tsv = args.custom_test_tsv
    else:
        _test_tsv = config["test_tsv"]
    eval_res = evaluate_model(model, cuda_device, _test_tsv, config["max_test_batch_count"], config, logger, vocabulary)
    
    test_rank_scores = eval_res["scores"]
    uncertainty = eval_res["uncertainty"]
    # pdb.set_trace()
    
    #
    # save full rerank results
    #
    prefix = ""
    if args.test_files_prefix:
        prefix = str(args.test_files_prefix)

    while os.path.isfile(os.path.join(run_folder, prefix + "test-run-full-rerank" + ".txt")): #this file already exists
        logger.info("WARNING | found existing test results in the folder, trying to save new results with different prefix")
        prefix += "new-" # adding shameful meaningless prefix for every existing file
    logger.info("Saving new files with prefix: " + prefix)

    test_file_path = os.path.join(run_folder, prefix + "test-run-full-rerank" + ".txt")
    save_sorted_results(test_rank_scores, test_file_path)

    #
    # compute evaluation metrics 
    # ---------------------------------
    #
    if reference_set_rank != None:
        result_info = compute_metrics_threshold_exact(evaluator, test_file_path,reference_set_rank, 
                                                      reference_set_tuple, reference_set_n, metric_tocompare=metric_tocompare)
    else:
        result_info = compute_metrics_from_files(evaluator, test_file_path)

    # save evaluated rank list
    save_sorted_results(result_info["rank_score"], os.path.join(run_folder, prefix + "test-run" + ".txt"))

    # save test info 
    result_info_tosave = {"cs@n": result_info["cs@n"], "metrics_avg": result_info["metrics_avg"],
                          "metrics_perq": result_info["metrics_perq"]}
    with open(os.path.join(run_folder, prefix + "test-metrics" + ".txt"), "w") as fw:
        fw.write("{'cs@n':%d, 'metrics_avg':%s}" % (result_info_tosave["cs@n"], result_info_tosave["metrics_avg"]))
    with open(os.path.join(run_folder, prefix + "test-metrics" + ".pkl"), "wb") as fw:
        pickle.dump(result_info_tosave, fw)
        
    if uncertainty != {}:
        with open(os.path.join(run_folder, prefix + "test-uncertainty" + ".txt"), "w") as fw:
            fw.write("q_id\tdoc_id\trel\tuncertainty\n")
            for q in uncertainty.keys():
                for d in uncertainty[q].keys():
                    fw.write("%d\t%s\t%f\t%s\n" % (q, d, uncertainty[q][d]["rel"], str(uncertainty[q][d]["uncertainty"])
                                                                                           .replace("[","")
                                                                                           .replace("]","")
                                                                                           .replace(" ","")))

    logger.info("Test %s results with re ranking at %d: %s" % (metric_tocompare, reference_set_n,
                                                               result_info_tosave["metrics_avg"][metric_tocompare]))
    return result_info
