import os
import sys
import subprocess
import pdb
import pickle
import argparse

import eval_scripts

MAX_MRR_RANK = 200

class EvaluationTool():
    def __init__(self):
        pass
    
    def evaluate(self, candidate, run_path_for_save, evalparam="-q", validaterun=False):
        pass
        
class EvaluationToolTrec(EvaluationTool):
    
    def __init__(self, trec_eval_path, qrel_path, 
                 trec_measures_param="-m ndcg -m ndcg_cut -m recall -m P -m map -m recip_rank"):
        self.trec_eval_path = trec_eval_path
        self.qrel_path = qrel_path
        self.trec_measures_param = trec_measures_param
        
    def run_command(self, command):
        #p = subprocess.Popen(command.split(),
        #                     stdout=subprocess.PIPE,
        #                     stderr=subprocess.STDOUT)
        #return iter(p.stdout.readline, b'')
        stdoutdata = subprocess.getoutput(command)
        return stdoutdata.split('\n')


    def validate_correct_runfile(self, run_path):
        run_path_temp = run_path+'.temp'
        with open(run_path) as fr, open(run_path_temp, 'w') as fw:
            qid = 0
            docids = set([])
            for l in fr:
                vals = [x.strip() for x in l.strip().split()]
                if vals[0] != qid:
                    qid = vals[0]
                    docids = set([])
                if vals[2] in docids:
                    continue
                docids.add(vals[2])
                fw.write("%s %s %s %s %s %s\n" % (vals[0], vals[1], vals[2], len(docids), vals[4], vals[5]))
        os.remove(run_path)
        os.rename(run_path_temp, run_path)


    def evaluate(self, candidate, run_path_for_save, evalparam="-q", validaterun=False):
        
        eval_scripts.save_sorted_results(candidate, run_path_for_save)
        results_avg, results_perq = self.evaluate_from_file(candidate_path=run_path_for_save, 
                                                            evalparam=evalparam, validaterun=validaterun)
        os.remove(run_path_for_save)
    
        return results_avg, results_perq
    
    def evaluate_from_file(self, candidate_path, evalparam="-q", validaterun=False):
        
        if validaterun:
            self.validate_correct_runfile(candidate_path)

        results_perq = {}
        results_avg = {}
        command = "%s %s %s %s %s" % (self.trec_eval_path, self.trec_measures_param, evalparam, self.qrel_path,
                                      candidate_path)
        print (command)
        for line in self.run_command(command):
            values = line.strip('\n').strip('\r').split('\t')
            if len(values) > 2:
                metric = values[0].strip()
                qid = values[1].strip()
                res = float(values[2].strip())
                if qid == 'all':
                    if metric not in results_avg:
                        results_avg[metric] = {}
                    results_avg[metric] = res
                else:
                    if metric not in results_perq:
                        results_perq[metric] = {}
                    results_perq[metric][qid] = res
                    
        return results_avg, results_perq

    
class EvaluationToolMsmarco(EvaluationTool):
    
    def __init__(self, qrel_path):
        self.qrel_path = qrel_path
        self.qids_to_relevant_docids = self.load_reference(self.qrel_path)
        
    def load_reference_from_stream(self, f):
        """Load Reference reference relevant documents
        Args:f (stream): stream to load.
        Returns:qids_to_relevant_docids (dict): dictionary mapping from query_id (int) to relevant documents (list of ints). 
        """
        qids_to_relevant_docids = {}
        for l in f:
            vals = l.strip().split('\t')
            if len(vals) != 4:
                vals = l.strip().split(' ')
                if len(vals) != 4:
                    pdb.set_trace()
                    raise IOError('\"%s\" is not valid format' % l)

            qid = vals[0]
            if qid in qids_to_relevant_docids:
                pass
            else:
                qids_to_relevant_docids[qid] = []
            _rel = int(vals[3])
            if _rel > 0:
                qids_to_relevant_docids[qid].append(vals[2])

        return qids_to_relevant_docids

    def load_reference(self, path_to_reference):
        """Load Reference reference relevant documents
        Args:path_to_reference (str): path to a file to load.
        Returns:qids_to_relevant_docids (dict): dictionary mapping from query_id (int) to relevant documents (list of ints). 
        """
        with open(path_to_reference,'r') as f:
            qids_to_relevant_docids = self.load_reference_from_stream(f)
        return qids_to_relevant_docids

    def evaluate(self, candidate, run_path_for_save, evalparam=None, validaterun=False):
        
        """Compute MRR metric
        """
        results_perq = {'recip_rank':{}}
        results_avg = {'recip_rank':{}}
        MRR = 0
        ranking = []
        for qid in candidate:
            if qid in self.qids_to_relevant_docids:
                target_docid = self.qids_to_relevant_docids[qid]
                candidate_docidscore = list(candidate[qid].items())
                candidate_docidscore.sort(key=lambda x: x[1], reverse=True)
                candidate_docid = [x[0] for x in candidate_docidscore]
                ranking.append(0)
                results_perq['recip_rank'][qid] = 0
                # MRR
                for i in range(0, min(len(candidate_docid), MAX_MRR_RANK)):
                    if candidate_docid[i] in target_docid:
                        MRR += 1/(i + 1)
                        results_perq['recip_rank'][qid] = 1/(i + 1)
                        ranking.pop()
                        ranking.append(i+1)
                        break
                
        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

        results_avg['recip_rank'] = MRR/len(ranking)
        
        return results_avg, results_perq

