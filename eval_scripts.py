"""
adaptod from https://github.com/dfcf93/MSMARCOV2/blob/master/Ranking/Evaluation/msmarco_eval.py
"""
import sys
import pdb
import statistics
import copy
from collections import Counter
import numpy as np
#import pytrec_eval 
import pickle
import os
import time

METRICS = {'map', 'ndcg_cut', 'recip_rank', 'P', 'recall'}

def load_reference_from_stream(f):
    """Load Reference reference relevant documents
    """
    qids_to_relevant_docids = {}
    for l in f:
        vals = l.strip().split('\t')
        if len(vals) != 4:
            vals = l.strip().split(' ')
            if len(vals) != 4:
                raise IOError('\"%s\" is not valid format' % l)

        #qid = int(vals[0])
        qid = vals[0]
        docid = vals[2]
        score = int(vals[3])
        if qid not in qids_to_relevant_docids:
            qids_to_relevant_docids[qid] = {}
        if docid in qids_to_relevant_docids[qid]:
            raise Error("One doc can not have multiple relevance for a query. QID=%d, docid=%s" % (qid, docid))
        qids_to_relevant_docids[qid][docid] = score

    return qids_to_relevant_docids

def load_reference(path_to_reference):
    """Load Reference reference relevant documents
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_docids = load_reference_from_stream(f)
    return qids_to_relevant_docids

def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    """
    qid_to_ranked_candidate_docs = {}
    for l in f:
            l = l.strip().split()
            try:
                if len(l) == 4: # own format
                    #qid = int(l[0])
                    qid = l[0]
                    docid = l[1]
                    rank = int(l[2])
                    score = float(l[3])
                if len(l) == 6: # original trec format
                    #qid = int(l[0])
                    qid = l[0]
                    docid = l[2]
                    rank = int(l[3])
                    score = float(l[4])
            except:
                raise IOError('\"%s\" is not valid format' % l)
            if qid not in qid_to_ranked_candidate_docs:
                qid_to_ranked_candidate_docs[qid] = {}
            if docid in qid_to_ranked_candidate_docs[qid]:
                raise Exception("Cannot rank a doc multiple times for a query. QID=%s, docid=%s" % (str(qid), str(docid)))
            qid_to_ranked_candidate_docs[qid][docid] = score
    return qid_to_ranked_candidate_docs

def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    """

    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_docs = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_docs

def compute_metrics(evaluator, qids_to_ranked_candidate, candidate_path_for_save, save_perquery=True):
    
    # code for pytrec_eval library
    #results_perq = evaluator.evaluate(qids_to_ranked_candidate_docs)
    #_metrics = list(list(results_perq.values())[0].keys())
    #for _metric in _metrics:
    #    results_avg[_metric] = pytrec_eval.compute_aggregated_measure(_metric, [x[_metric] for x in results_perq.values()])
    
    #code for adhoc trec_eval
    _evalparam = "-q" if save_perquery else ""
    results_avg, results_perq = evaluator.evaluate(candidate=qids_to_ranked_candidate, 
                                                   run_path_for_save=candidate_path_for_save,
                                                   evalparam=_evalparam)
    
    return results_avg, results_perq

def compute_metrics_from_files(evaluator, path_to_candidate):
    qids_to_ranked_candidate_docs = load_candidate(path_to_candidate)
    
    results_avg, results_perq = evaluator.evaluate_from_file(candidate_path=path_to_candidate)
    
    result_info = {}
    result_info["metrics_avg"] = results_avg
    result_info["metrics_perq"] = results_perq
    result_info["rank_score"] = qids_to_ranked_candidate_docs
    result_info["cs@n"] = -1

    return result_info

def compute_metrics_threshold_range(evaluator, path_to_candidate, reference_set_rank, reference_set_tuple, 
                                    candidate_from_to, candidate_interval, metric_tocompare):
    
    qids_to_ranked_candidate_docs = load_candidate(path_to_candidate)
    #evaluator = pytrec_eval.RelevanceEvaluator(qids_to_relevant_docids, METRICS)
    
    metric_results_avg = {}
    best_result = 0
    best_result_info = {}
    
    #start_time = time.time()
    #elapsed_time_calc = start_time - start_time
    #elapsed_time_save = start_time - start_time
    #elapsed_time_eval = start_time - start_time
        
    for i in range(candidate_from_to[0], candidate_from_to[1] + 1, candidate_interval):

        pruned_qids_to_ranked_candidate_docs = {}

        #start_time = time.time()
        
        for qid in qids_to_ranked_candidate_docs:
            rank_docids = list(qids_to_ranked_candidate_docs[qid].items())
            rank_docids.sort(key=lambda x: x[1], reverse=True)
            rank_docids = [x[0] for x in rank_docids]

            pruned_qids_to_ranked_candidate_docs[qid] = {}
            added = 0
            added_docids = []
            for rank, docid in enumerate(rank_docids):
                if docid in reference_set_rank[qid] and reference_set_rank[qid][docid] <= i:
                    pruned_qids_to_ranked_candidate_docs[qid][docid] = qids_to_ranked_candidate_docs[qid][docid]
                    added_docids.append(docid) 
                    added += 1

            _scores = list(pruned_qids_to_ranked_candidate_docs[qid].values())
            if len(_scores) > 0:
                score_diff = np.min(_scores) - np.max([x[1] for x in reference_set_tuple[qid]])
            else:
                score_diff = np.max([x[1] for x in reference_set_tuple[qid]])
            # adding what is rest till i to the list if it is missing
            for _tuple in reference_set_tuple[qid]:
                docid, score = _tuple
                if i <= added:
                    break
                if docid not in added_docids:
                    pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff # make scores lower than "pruned"
                    added += 1

            # adding the rest from i or added till the end
            for j in range(added, len(reference_set_tuple[qid])):
                docid, score = reference_set_tuple[qid][j]
                pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff

        candidate_path_for_save_pruned = path_to_candidate+'.pruned'
        #path_to_candidate_pruned = path_to_candidate+'.pruned'
        #save_sorted_results(pruned_qids_to_ranked_candidate_docs, path_to_candidate_pruned)
        
        metric_results_avg[i], metric_results_perq = compute_metrics(evaluator, pruned_qids_to_ranked_candidate_docs,
                                                                     candidate_path_for_save_pruned)
        
        if metric_results_avg[i][metric_tocompare] > best_result:
            best_result = metric_results_avg[i][metric_tocompare]
            best_result_info["metrics_avg"] = metric_results_avg[i]
            best_result_info["metrics_perq"] = metric_results_perq
            best_result_info["rank_score"] = pruned_qids_to_ranked_candidate_docs
            best_result_info["cs@n"] = i
    
    return best_result_info, metric_results_avg

def compute_metrics_threshold_exact(evaluator, path_to_candidate, reference_set_rank, reference_set_tuple,
                                    candidate_threshold, metric_tocompare):

    qids_to_ranked_candidate_docs = load_candidate(path_to_candidate)
    
    pruned_qids_to_ranked_candidate_docs = {}
    for qid in qids_to_ranked_candidate_docs:
        rank_docids = list(qids_to_ranked_candidate_docs[qid].items())
        rank_docids.sort(key=lambda x: x[1], reverse=True)
        rank_docids = [x[0] for x in rank_docids]

        pruned_qids_to_ranked_candidate_docs[qid] = {}
        added = 0
        added_docids = []
        for rank, docid in enumerate(rank_docids):
            if docid in reference_set_rank[qid] and reference_set_rank[qid][docid] <= candidate_threshold:
                pruned_qids_to_ranked_candidate_docs[qid][docid] = qids_to_ranked_candidate_docs[qid][docid]
                added_docids.append(docid)
                added += 1

        _scores = list(pruned_qids_to_ranked_candidate_docs[qid].values())
        if len(_scores) > 0:
            score_diff = np.min(_scores) - np.max([x[1] for x in reference_set_tuple[qid]])
        else:
            score_diff = np.max([x[1] for x in reference_set_tuple[qid]])
        # adding what is rest till i to the list if it is missing
        for _tuple in reference_set_tuple[qid]:
            docid, score = _tuple
            if candidate_threshold <= added:
                break
            if docid not in added_docids:
                pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff # make scores lower than "pruned"
                added += 1

        # adding the rest from i or added till the end
        for j in range(added, len(reference_set_tuple[qid])):
            docid, score = reference_set_tuple[qid][j]
            pruned_qids_to_ranked_candidate_docs[qid][docid] = score + score_diff

    candidate_path_for_save_pruned = path_to_candidate+'.pruned'
    #path_to_candidate_pruned = path_to_candidate+'.pruned'
    #save_sorted_results(pruned_qids_to_ranked_candidate_docs, path_to_candidate_pruned)
    metric_results_avg, metric_results_perq = compute_metrics(evaluator, pruned_qids_to_ranked_candidate_docs,
                                                              candidate_path_for_save_pruned)
        
    result_info = {}
    result_info["metrics_avg"] = metric_results_avg
    result_info["metrics_perq"] = metric_results_perq
    result_info["rank_score"] = pruned_qids_to_ranked_candidate_docs
    result_info["cs@n"] = candidate_threshold

    return result_info


def save_sorted_results(results, file, until_rank=-1):
    with open(file, "w") as val_file:
        for qid in results.keys():
            query_data = list(results[qid].items())
            query_data.sort(key=lambda x: x[1], reverse=True)
            # sort the results per query based on the output
            for rank_i, (docid, score) in enumerate(query_data):
                #val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value])+"\n")
                val_file.write("%s Q0 %s %d %f neural\n" % (str(qid), str(docid), rank_i + 1, score))

                if until_rank > -1 and rank_i == until_rank + 1:
                    break
