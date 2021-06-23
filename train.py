#
# train a neural-ir model
# -------------------------------
#
# features:
#
# * uses pytorch + allenNLP
# * tries to correctly encapsulate the data source (msmarco)
# * clear configuration with yaml files
#
# usage:
# python train.py --run-name experiment1 --config-file configs/knrm.yaml


import argparse
import copy
import os
import pdb
#import GPUtil
import pickle
import glob
import time
import sys
import numpy as np
sys.path.append(os.getcwd())
from typing import Dict, Tuple, List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torch.multiprocessing as mp

from allennlp.common import Params, Tqdm
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import move_to_device
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from transformers import BertModel, BertConfig, AutoConfig, BartForConditionalGeneration, BartConfig

from tensorboardX import SummaryWriter
#import pytrec_eval

import getpass
username = getpass.getuser()
#sys.stdout.write("User name '%s' is running the program\n" % username)
#sys.stdout.flush()
#if username in ["navid"]:
#    sys.path.append("/home/navid/rk0/home/navid/repository/neuroir_works")
    
from utils import *
from optimizers import Optimizer
from models.matching_knrm import *
from models.matching_conv_knrm import *
from models.matching_matchpyramid import *
from models.matching_pacrr import *
from models.matching_tk import *
from models.matching_bert import *
from models.matching_bertkernel import *

from models.generative_seq2seqatt import *
from models.generative_pgn import *
from models.generative_transpgn import *
from models.generative_trans2trans import *
from models.generative_bert2trans import *
from models.generative_bart import *

from eval_models import *
from eval_scripts import *
from multiprocess_input_pipeline import *
from eval_evaluation_tool import EvaluationToolTrec, EvaluationToolMsmarco

Tqdm.default_mininterval = 1

EPS = 1e-20



# label is always set to 1 - 
# for MarginRankingLoss: it indicates that the first input is positive
def get_label(batch_size, cuda_device):
    label = torch.ones(batch_size)
    if cuda_device != -1:
        label = label.cuda(cuda_device)
    return label

def evaluate_validation():
    global best_result_info
    
    # evaluate validation set -> extend to test set as well
    _best_result_info = validate_model(model, config, logger, run_folder, cuda_device, writer=writer,
                                       epoch_number=epoch, batch_cnt_global=batch_cnt_global, 
                                       metric_tocompare=config["metric_tocompare"],
                                       evaluator=evaluator_val,
                                       vocabulary=vocab,
                                       reference_set_rank=reference_set_rank_val, 
                                       reference_set_tuple=reference_set_tuple_val,
                                       batch_number=i, global_best_info=best_result_info)
    
    #writer.add_scalar("validation/%s" % config["metric_tocompare"], 
    #                  _best_result_info["metrics_avg"][config["metric_tocompare"]], batch_cnt_global)
    if (best_result_info is None) or (_best_result_info["metrics_avg"][config["metric_tocompare"]] > 
                                      best_result_info["metrics_avg"][config["metric_tocompare"]]):
        best_result_info = _best_result_info
        
        # save best results
        best_validation_output_path = os.path.join(run_folder, "best-validation-run.txt")
        save_sorted_results(best_result_info["rank_score"], best_validation_output_path)    

        best_result_info_tosave = {"cs@n": best_result_info["cs@n"],
                                   "metrics_avg": best_result_info["metrics_avg"],
                                   "metrics_perq": best_result_info["metrics_perq"],
                                   "epoch": best_result_info["epoch"],
                                   "batch_number": best_result_info["batch_number"]}
        with open(best_result_info_path, "w") as fw:
            fw.write("{'cs@n':%d, 'metrics_avg':%s, 'epoch':%d, 'batch_number':%d}" % 
                     (best_result_info_tosave["cs@n"], str(best_result_info_tosave["metrics_avg"]),
                      best_result_info_tosave["epoch"], best_result_info_tosave["batch_number"]))
        with open(best_result_info_pkl_path, "wb") as fw:
            pickle.dump(best_result_info_tosave, fw)
        
        # save best model state
        logger.info("Saving new model with %s of %.4f at %s" % (config["metric_tocompare"], 
                                                                best_result_info['metrics_avg'][config["metric_tocompare"]],
                                                                best_model_store_path))
        #writer.add_text("checkpoint", "saving best model at epoch %3d and %5d batches" % (epoch, i), batch_cnt_global)
        model_save(best_model_store_path, model, best_result_info)

    return _best_result_info
        
#
# main process
# -------------------------------
#
if __name__ == "__main__":
    
    ###############################################################################
    # Inititialization
    ###############################################################################

    #
    # config
    # -------------------------------
    #
    args = get_parser().parse_args()

    run_folder, config = prepare_experiment(args)
    
    ## temporary code for old experiment
    if "metric_tocompare" not in config:
        config["metric_tocompare"] = "MRR"
    ##-----

    writer = SummaryWriter(run_folder)

    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    logger.info("Experiment folder: %s", run_folder)
    logger.info("Config vars: %s", str(config))

    #writer.add_text("info", "Config vars: %s" % str(config))

    # Set the random seed manually for reproducibility.
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("You have a CUDA device, so you should probably run with --cuda")
            #writer.add_text("warning", "You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed_all(config["seed"])


    if args.cuda:
        # hardcode gpu usage
        cuda_device = int(args.cuda_device_id)
        torch.cuda.init() # just to make sure we can select a device
        torch.cuda.set_device(cuda_device)
        logger.info("Using cuda device id: %i - %s", cuda_device, torch.cuda.get_device_name(cuda_device))
    else:
        cuda_device = -1


    #
    # create (or load) model instance
    # -------------------------------
    #
    # * vocab (pre-built, to make the embedding matrix smaller, see generate_vocab.py)
    # * pre-trained embedding
    # * network
    # * optimizer & loss function
    #

    #
    # load candidate set for efficient cs@N validation
    #
    reference_set_rank_val, reference_set_tuple_val = parse_reference_set(config["validation_candidate_set_path"],
                                                                          config["validation_candidate_set_from_to"][1])
    #qids_to_relevant_docids_val = load_reference(path_to_reference=config["validation_qrels"])
    #evaluator_val = pytrec_eval.RelevanceEvaluator(qids_to_relevant_docids_val, METRICS)
    
    
    if config["metric_tocompare"] == 'recip_rank':
        evaluator_val = EvaluationToolMsmarco(qrel_path=config["validation_qrels"])
    else:
        _val_trec_measures_param = "-m %s" % config["metric_tocompare"]
        evaluator_val = EvaluationToolTrec(trec_eval_path=config["trec_eval_path"], qrel_path=config["validation_qrels"],
                                           trec_measures_param=_val_trec_measures_param)

    ###############################################################################
    # Load data 
    ###############################################################################

    if config["token_embedder_type"] in ["embedding", "random", "elmo"]:
        logger.info("Loading vocabularies")
        vocab = Vocabulary.from_files(config["vocab_directory"])
        vocab.add_tokens_to_namespace([START_SYMBOL, END_SYMBOL])
        logger.info("Vocabulary size %d" % vocab.get_vocab_size())
        #writer.add_text("info", "Vocabulary size %d" % vocab.get_vocab_size(), 0)

        # embedding layer (use pre-trained, but make it trainable as well)
        if config["token_embedder_type"] == "embedding":
            logger.info("Loading embedding %s" % config["pre_trained_embedding"])
            tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                                  "embedding_dim": config["pre_trained_embedding_dim"],
                                                                  "trainable": config["train_embedding"],
                                                                  "padding_index":0}))
        elif config["token_embedder_type"] == "random":
            logger.info("Initializing random embedding")
            tokens_embedder = Embedding.from_params(vocab, Params({"embedding_dim":config["pre_trained_embedding_dim"],
                                                                   "trainable": config["train_embedding"],
                                                                   "padding_index":0}))
        elif config["token_embedder_type"] == "elmo":
            logger.info("Loading ELMO")
            tokens_embedder = ElmoTokenEmbedder(config["elmo_options_file"], config["elmo_weights_file"])
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
    elif config["token_embedder_type"] in ["bert", "bart"]:
        vocab = None
        
        if config["token_embedder_type"] == "bert":
            logger.info("Loading BERT")
            if config["transformers_pretrained_model_id"] == "random":
                _bert_config_standard = AutoConfig.for_model('bert')
                bert_config = BertConfig(vocab_size = _bert_config_standard.vocab_size,
                                         hidden_size = config["bert_hidden_size"], #?
                                         num_hidden_layers = config["bert_num_layers"],
                                         num_attention_heads = config["bert_num_heads"],
                                         intermediate_size = config["bert_intermediate_size"],
                                         hidden_dropout_prob = config["bert_dropout"], #!
                                         attention_probs_dropout_prob = config["bert_dropout"])
                transformer_embedder = BertModel(bert_config)
            else:
                transformer_embedder = BertModel.from_pretrained(config["transformers_pretrained_model_id"], cache_dir="cache")
        elif config["token_embedder_type"] == "bart":
            logger.info("Loading BART")
            transformer_embedder = BartForConditionalGeneration.from_pretrained(config["transformers_pretrained_model_id"], cache_dir="cache")
    else:
        logger.error("token_embedder_type %s not known",config["token_embedder_type"])
        exit(1)

    ###############################################################################
    # Model
    ###############################################################################

    if config["model"] == "knrm":
        model = KNRM(word_embedder, vocab, n_kernels=config["knrm_kernels"])
    elif config["model"] == "conv_knrm":
        model = Conv_KNRM(word_embedder, vocab, n_kernels=config["conv_knrm_kernels"],
                          conv_out_dim=config["conv_knrm_conv_out_dim"])
    elif config["model"] == "match_pyramid":
        model = MatchPyramid(word_embedder, vocab, 
                             conv_output_size=config["match_pyramid_conv_output_size"],
                             conv_kernel_size=config["match_pyramid_conv_kernel_size"], 
                             adaptive_pooling_size=config["match_pyramid_adaptive_pooling_size"])
    elif config["model"] == "pacrr":
        model = PACRR(word_embedder, vocab, unified_query_length=config["pacrr_unified_query_length"],
                      unified_document_length=config["pacrr_unified_document_length"], 
                      max_conv_kernel_size=config["pacrr_max_conv_kernel_size"], 
                      conv_output_size=config["pacrr_conv_output_size"], 
                      kmax_pooling_size=config["pacrr_kmax_pooling_size"],
                      idfcf_path=config["idfcf_path"])
    elif config["model"] == "tk":
        model = TK(word_embeddings=word_embedder,
                   vocab=vocab, 
                   kernels_mu=config["tk_kernels_mu"],
                   kernels_sigma=config["tk_kernels_sigma"], 
                   att_heads=config["tk_att_heads"],
                   att_layer=config["tk_att_layer"],
                   att_intermediate_size=config["tk_att_intermediate_size"])
    elif config["model"] == "bertkernel":
        model = BertKernel(bert = transformer_embedder, 
                           kernels_mu=config["tk_kernels_mu"],
                           kernels_sigma=config["tk_kernels_sigma"])
    elif config["model"] == "discbert":
        model = DiscriminativeBert(bert = transformer_embedder)
        _bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # check the way tokenization works
    
    elif config["model"] == "seq2seqatt":
        model = SeqToSeqWithAttention(vocab, word_embedder, 
                                      lstm_hidden_dim = config["seq2seq_lstm_hidden_dim"],
                                      enc_lstm_num_layers = config["seq2seq_enc_lstm_num_layers"],
                                      projection_dim = config["seq2seq_projection_dim"],
                                      tie = config["seq2seq_tie"])
    elif config["model"] == "pgn":
        model = PointerGeneratorNetwork(vocab, word_embedder, 
                                        lstm_hidden_dim = config["pgn_lstm_hidden_dim"],
                                        enc_lstm_num_layers = config["pgn_enc_lstm_num_layers"],
                                        projection_dim = config["pgn_projection_dim"],
                                        tie = config["pgn_tie"])
    elif config["model"] == "pgnt":
        model = TransformerPointerGeneratorNetwork(vocab, word_embedder, 
                                        lstm_hidden_dim = config["tpgn_lstm_hidden_dim"],
                                        enc_lstm_num_layers = config["tpgn_enc_lstm_num_layers"],
                                        enc_transf_num_layers = config["tpgn_enc_trans_num_layers"],
                                        enc_transf_intermediate_size = config["tpgn_enc_trans_intermediate_size"],
                                        enc_transf_num_heads = config["tpgn_enc_trans_num_heads"],
                                        enc_transf_dropout = config["tpgn_enc_trans_dropout"],
                                        projection_dim = config["tpgn_projection_dim"],
                                        tie = config["tpgn_tie"])
    elif config["model"] == "t2t":
        model = SeqToSeqTransformers(vocab, word_embedder,
                                     encoder_num_layers = config["t2t_encoder_num_layers"],
                                     decoder_num_layers = config["t2t_decoder_num_layers"],
                                     transf_intermediate_size = config["t2t_intermediate_size"],
                                     num_heads = config["t2t_num_heads"],
                                     dropout = config["t2t_dropout"],
                                     projection_dim = config["t2t_output_proj_dim"],
                                     tie = config["t2t_tie"])
    elif config["model"] == "bert2t":
        model = BERT2Transformer(encoder_bert = transformer_embedder, 
                             decoder_hidden_size = config["bert2t_decoder_hidden_size"],
                             decoder_num_layers = config["bert2t_decoder_num_layers"],
                             decoder_intermediate_size = config["bert2t_decoder_intermediate_size"],
                             decoder_num_heads = config["bert2t_decoder_num_heads"],
                             decoder_dropout = config["bert2t_decoder_dropout"],
                             tie_encoder_decoder = config["bert2t_tie_encoder_decoder"],
                             tie_decoder_output = config["bert2t_tie_decoder_output"])
    elif config["model"] == "bart":
        model = Bart(bartmodel = transformer_embedder)
    
    else:
        logger.error("Model %s not known", config["model"])
        exit(1)

    if cuda_device != -1:
        model.cuda(cuda_device)

    logger.info('Model: %s', model)
    logger.info('Model total parameters: %s', sum(p.numel() for p in model.parameters()))
    logger.info('Model total trainable parameters: %s', sum(p.numel() for p in model.parameters() if p.requires_grad))

    #writer.add_text("model", "Model: %s" % str(model))
    #writer.add_text("model", "Total parameters: %s" % (sum(p.numel() for p in model.parameters())))
    #writer.add_text("model", "Total trainable parameters: %s" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    if config["optimizer"] == "adam":
        params_group0 = []
        params_group1 = []
        
        for p_name, par in model.named_parameters():
            spl = p_name.split(".")
            
            group1_check = p_name
            if len(spl) >= 2:
                group1_check = spl[0]+"."+spl[1]
            #print(group1_check)

            if group1_check in config["param_group1_names"]:
                params_group1.append(par)
                logger.info("params_group1: %s", p_name)
            else:
                params_group0.append(par)
        
        all_params =[{"params":params_group0, "lr":config["param_group0_learning_rate"], "weight_decay":config["param_group0_weight_decay"]}, {"params":params_group1, "lr":config["param_group1_learning_rate"], "weight_decay":config["param_group1_weight_decay"]}
        ]

        optimizer = optim.Adam(all_params, betas=(0.9, 0.999), eps=0.00001)
        
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                     patience=config["learning_rate_scheduler_patience"],
                                     factor=config["learning_rate_scheduler_factor"],
                                     verbose=True)
    early_stopper = EarlyStopping(patience=config["early_stopping_patience"], mode="max", min_delta=0.0005)   
    
    
    if config["loss"] in ["maxmargin"]:
        criterion = torch.nn.MarginRankingLoss(margin=config["loss_maxmargin_margin"], reduction='mean')
    else:
        criterion = None

    if (cuda_device != -1) and (criterion is not None):
        criterion.cuda(cuda_device)
                    
    ###############################################################################
    # Train and Validation
    ###############################################################################
    best_result_info_path = os.path.join(run_folder, "best-validation-metrics.txt")
    best_result_info_pkl_path = os.path.join(run_folder, "best-validation-metrics.pkl")
    best_model_store_path = os.path.join(run_folder, "model.best.pt")
    checkpoint_model_store_path = os.path.join(run_folder, "model.checkpoint.pt")


    if not args.test:
        #TODO implementing continuos training args.continue_train
        
        logger.info('-' * 89)
        logger.info('Training started...')
        logger.info('-' * 89)

        # keep track of the best metric
        best_result_info = None 
        
        #
        # training / saving / validation loop
        # -------------------------------
        #

        training_batch_size = int(config["batch_size_train"])
        validate_every_n_batches = config["validate_every_n_batches"]
        # helper vars for quick checking if we should validate during the epoch
        do_validate_every_n_batches = validate_every_n_batches > -1
        loss_sum = 0
        data_cnt_all = 0
        training_processes = []
        batch_cnt_global = 0

        try:
            for epoch in range(0, int(config["epochs"])):
                if early_stopper.stop:
                    break
                #
                # data loading
                # -------------------------------
                #
                train_files = glob.glob(config.get("train_tsv"))
                training_queue, training_processes, train_exit = get_multiprocess_batch_queue("train-batches-" + str(epoch),
                                                                                              multiprocess_training_loader,
                                                                                              files=train_files,
                                                                                              conf=config,
                                                                                              _logger=logger,
                                                                                              vocabulary=vocab)
                
                model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...
                #time.sleep(len(training_processes))  # fill the queue
                logger.info("[Epoch %d] --- Start training with queue.size:%d" % (epoch, training_queue.qsize()))
                label = get_label(training_batch_size, cuda_device)

                #
                # train loop
                # -------------------------------
                #
                #for i in Tqdm.tqdm(range(0, config["training_batch_count"]), disable=config["tqdm_disabled"]):
                i = 0
                batch_null_cnt = 0
                max_training_batch_count = config["max_training_batch_count"]
                #writer.add_text("epochs", "epoch %3d started" % epoch, batch_cnt_global)
                while (True):
                    batch = training_queue.get()
                    if batch is None:
                        batch_null_cnt += 1
                        if batch_null_cnt == len(train_files):
                            break
                        else:
                            continue
                    if i >= max_training_batch_count and max_training_batch_count != -1:
                        break
                    batch_cnt_global += 1

                    if config["token_embedder_type"] in ["bert", "bart"]:
                        # adapting ir_triple_transformers_loader format to allennlp format used in standard loaders;
                        batch = {"query_tokens" : {"tokens" : batch["query_tokens"].long()},
                                 "doc_pos_tokens" : {"tokens" : batch["doc_pos_tokens"].long()},
                                 "doc_neg_tokens" : {"tokens" : batch["doc_neg_tokens"].long()}}
                    
                    if isinstance(batch["query_tokens"], dict):
                        current_batch_size = int(batch["query_tokens"]["tokens"].shape[0])
                    else:
                        current_batch_size = int(batch["query_tokens"].shape[0])

                    if cuda_device != -1:
                        batch = move_to_device(batch, cuda_device)

                    optimizer.zero_grad()

                    # output for the positive document
                    if config["model"].startswith("pgn"):
                        # PGN requires additional inputs;
                        ### local ditionaries are needed to allow for differentiation between multiple
                        ### unknown words in the input (words not from our vocabulary)
                        output_pos_dict = model.forward(target = batch["query_tokens"],
                                                   target_local_token_ids = batch["query_local_dict"],
                                                   source = batch["doc_pos_tokens"],
                                                   source_local_token_ids = batch["doc_pos_local_dict"])
                    else:
                        output_pos_dict = model.forward(batch["query_tokens"], batch["doc_pos_tokens"])
                    
                    if config["loss"] != "negl":
                        # output for the negative document
                        if config["model"].startswith("pgn"):
                            output_neg_dict = model.forward(target = batch["query_tokens"],
                                                       target_local_token_ids = batch["query_local_dict"],
                                                       source = batch["doc_neg_tokens"],
                                                       source_local_token_ids = batch["doc_neg_local_dict"])         
                        else:
                            output_neg_dict = model.forward(batch["query_tokens"], batch["doc_neg_tokens"])#, unlikely = True
                        output_neg = output_neg_dict["rels"]
                    output_pos = output_pos_dict["rels"]

                    # this only affect the last n batches (n = # data loader processes)
                    if current_batch_size != training_batch_size:
                        label = get_label(current_batch_size, cuda_device)
                    data_cnt_all += current_batch_size

                    # GPUtil.showUtilization()
                    # loss
                    loss = None 
                    if config["loss"] == "negl":
                        loss = - torch.mean(output_pos)
                    elif config["loss"] == "lul":
                        #pdb.set_trace()
                        outputs = output_pos + torch.log(1 - torch.exp(output_neg) + EPS)
                        loss = - torch.mean(outputs)
                        #loss = - (torch.mean(output_pos) + torch.mean(output_neg))
                    elif config["loss"] == "maxmargin":
                        loss = criterion(output_pos, output_neg, label)
                    elif config["loss"] == "maxmarginexp":
                        loss = criterion(torch.exp(output_pos), torch.exp(output_neg), label)
                    elif config["loss"] == "crossentropy":
                        outputs = output_pos_dict["logprobs"][:,0] + output_neg_dict["logprobs"][:,1]
                        loss = - torch.mean(outputs)
                        
                    # GPUtil.showUtilization()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    # GPUtil.showUtilization()

                    # set the label back to a cached version (for next iterations)
                    if current_batch_size != training_batch_size:
                        # GPUtil.showUtilization()
                        label = get_label(training_batch_size, cuda_device)

                    loss_sum += loss.item()
                    
                    # only needed for profiling to get the remainder of the cuda work done and not put into another line
                    #torch.cuda.synchronize()

                    #writer.add_scalar("loss", loss.item(), batch_cnt_global)
                    
                    if (i % validate_every_n_batches == 0):
                        logger.info('Experiment %s' % (args.run_name))

                    if (i % config["log_interval"] == 0) and (i != 0):
                        #logging
                        #GPUtil.showUtilization()
                        cur_loss = loss_sum / float(data_cnt_all)
                        
                        logger.info('| TRAIN | epoch %3d | %5d batches | lrs %s | avg. train loss %.5f' %
                                    (epoch, i, str(['%02.6f' % pg['lr'] for pg in optimizer.param_groups]), cur_loss))

                        # make sure that the perf of the queue is sustained
                        if training_queue.qsize() < 10:
                            logger.warning("training_queue.qsize() < 10 (%d)" % training_queue.qsize())
                        
                    if config["checkpoint_interval"] != -1 and i % config["checkpoint_interval"] == 0 and i > 0:
                        logger.info("saving checkpoint at epoch %3d and %5d batches" % (epoch, i))
                        #writer.add_text("checkpoint", "saving checkpoint at epoch %3d and %5d batches" % (epoch, i),
                        #                batch_cnt_global)
                        checkpoint_save(checkpoint_model_store_path, model, criterion, optimizer, epoch, i)

                    #
                    # validation (inside epoch) - if so configured
                    #
                    if do_validate_every_n_batches:
                        if i > 0 and i % validate_every_n_batches == 0:
                            logger.info('Experiment %s' % (args.run_name))
                            _result_info = evaluate_validation()
                            logger.info('Experiment %s' % (args.run_name))

                            # coming back to training mode
                            model.train()
                            if lr_scheduler is not None:
                                lr_scheduler.step(_result_info["metrics_avg"][config["metric_tocompare"]])
                            if early_stopper.step(_result_info["metrics_avg"][config["metric_tocompare"]]):
                                logger.info("early stopping epoch %d batch count %d" % (epoch, i))
                                #writer.add_text("earlystop", "early stopping at epoch %3d and %5d batches" % (epoch, i),
                                #                batch_cnt_global)
                                break
                            
                    i += 1 #next batch

                #writer.add_text("epochs", "epoch %3d finished with %5d batches" % (epoch, i), batch_cnt_global)

                # make sure we didn't make a mistake in the configuration / data preparation
                if training_queue.qsize() != 0 and config["max_training_batch_count"] == -1:
                    logger.error("training_queue.qsize() is not empty after epoch "+str(epoch))

                #logging loss
                cur_loss = loss_sum / float(data_cnt_all)
                logger.info('| TRAIN | epoch %3d FINISHED | %5d batches | lrs %s | avg. train loss %.5f' %
                            (epoch, i, str(['%02.6f' % pg['lr'] for pg in optimizer.param_groups]), cur_loss))
                if config["checkpoint_interval"] != -1:
                    logger.info("saving checkpoint at epoch %d after %d batches" % (epoch, i))
                    #writer.add_text("checkpoint", "saving checkpoint at epoch %3d and %5d batches" % (epoch, i), batch_cnt_global)
                    checkpoint_save(checkpoint_model_store_path, model, criterion, optimizer, epoch, i)

                #terminating sub processes
                train_exit.set()  # allow sub-processes to exit
                for proc in training_processes:
                    if proc.is_alive():
                        proc.terminate()

                #
                # validation (at the end of epoch)
                #
                _result_info = evaluate_validation()
                if lr_scheduler is not None:
                    lr_scheduler.step(_result_info["metrics_avg"][config["metric_tocompare"]])
                if early_stopper.step(_result_info["metrics_avg"][config["metric_tocompare"]]):
                    logger.info("early stopping epoch %d" % epoch)
                    #writer.add_text("earlystop", "early stopping at the end of epoch %3d" % epoch, batch_cnt_global)
                    break

        except Exception as e:
            logger.exception('[train] Got exception: %s' % str(e))
            logger.info('Exiting from training early')

            for proc in training_processes:
                if proc.is_alive():
                    proc.terminate()
            exit(1)

        logger.info('Training Finished!')
        logger.info('=' * 89)
      
    ###############################################################################
    # Test
    ###############################################################################
    #
    # evaluate the test set with the best model
    #
    logger.info('-' * 89)
    logger.info('Test set evaluation started...')
    logger.info('-' * 89)
    
    logger.info('Loading test qrels and reference set')
    if args.custom_test_candidates:
        logger.info("OVERRIDING: using custom test candidates: %s" % (args.custom_test_candidates))
        _test_candidate_set_path = args.custom_test_candidates
    else:
        _test_candidate_set_path = config["test_candidate_set_path"]
    reference_set_rank_test, reference_set_tuple_test = parse_reference_set(_test_candidate_set_path,
                                                                            config["test_candidate_set_max"])
    #qids_to_relevant_docids_test = load_reference(path_to_reference=config["test_qrels"])
    #evaluator_test = pytrec_eval.RelevanceEvaluator(qids_to_relevant_docids_test, METRICS)

    if args.custom_test_qrels:
        logger.info("OVERRIDING: using custom test_qrels: %s" % (args.custom_test_qrels))
        _test_qrels = args.custom_test_qrels
    else:
        _test_qrels = config["test_qrels"]
    evaluator_test = EvaluationToolTrec(trec_eval_path=config["trec_eval_path"], qrel_path=_test_qrels)
    
    logger.info('Loading best model')
    
    _cuda_device = int(args.cuda_device_id)
    model_state, best_result_info_val = model_load(best_model_store_path, _cuda_device)
    model.load_state_dict(model_state)
    
    logger.info('Testing the model with rerank at %d' % best_result_info_val["cs@n"])
    #writer.add_text("test", "Best rerank parameter: %d" % best_result_info_val["cs@n"])
    if args.custom_test_depth:
        logger.info("OVERRIDING: using custom depth for the test: %d" % (int(args.custom_test_depth)))
        depth = int(args.custom_test_depth)
    else:
        depth = best_result_info_val["cs@n"]
    test_result_info = test_model(args, model, config, logger, run_folder, cuda_device,
                                  metric_tocompare=config["metric_tocompare"],
                                  evaluator=evaluator_test,
                                  vocabulary=vocab,
                                  reference_set_rank=reference_set_rank_test, 
                                  reference_set_tuple=reference_set_tuple_test,
                                  reference_set_n=depth)

    #writer.add_text("test", "Test set %s: %.4f" % (config["metric_tocompare"], 
    #                                               test_result_info["metrics_avg"][config["metric_tocompare"]]))

    #writer.close()
    logger.info('Fertig!')
    
