#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

from exp_utils import create_exp_dir

from data_utils import FT_Dataset 
from model import GPT2Config, GPT2LMModel

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora.layer import Linear as peft_Linear

from sq1e.sen_qnn_ext import Qmodel_prep, senqnn_config_init, patch_torch_bmm
from sq1e.sen_qnn_infer import QLoRALinear

parser = argparse.ArgumentParser(description='PyTorch GPT2 beam decoding')

# add_gpu_params(parser)

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')

parser.add_argument('--seq_len', type=int, default=512,
                    help='number of tokens to predict')

parser.add_argument('--eval_len', type=int, default=256,
                    help='evaluation length')

parser.add_argument('--min_length', type=int, default=0,
                    help='minimum generation length')

parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--lora_dropout', default=0.0, type=float, help='dropout probability for lora layers')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder')

parser.add_argument('--beam', type=int, default=1, help='beam search size')

parser.add_argument('--length_penalty', type=float, default=1.0, help='length penalty')

parser.add_argument('--no_repeat_ngram_size', type=int, default=4, help='no_repeat_ngram_size')

parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')

parser.add_argument('--eos_token_id', action='append', type=int, default=[50256], 
                    help='eos token id')

parser.add_argument('--output_file', type=str, default='beam_prediction.jsonl', 
                    help='output file name')

parser.add_argument("--platform", default='k8s', type=str, help='platform cloud')

parser.add_argument("--random_seed", default=10, type=int, help='random seed')

parser.add_argument("--rank", default=0, type=int, help='rank')

parser.add_argument("--world_size", default=1, type=int, help='world size')

parser.add_argument("--local-rank", default=0, type=int, help='local rank')

parser.add_argument("--device", default=0, type=int, help='device')

# Add sq1e related parameters
parser.add_argument('--nbits_w', default=32, type=int, help='weight precision')
parser.add_argument('--nbits_a', default=32, type=int, help='activation precision')
parser.add_argument('--nbits_w_qkv', default=32, type=int, help='weight precision for qkv layers')
parser.add_argument('--nbits_a_qkv', default=32, type=int, help='weight precision for qkv layers')
parser.add_argument('--nbits_bmm1', default=32, type=int, help='weight precision for bmm1')
parser.add_argument('--nbits_bmm2', default=32, type=int, help='weight precision for bmm2')
parser.add_argument('--qw_mode', type=str, default='sawb', help='weight quantization, pick from lpuq, sawb or dorefa') 
parser.add_argument('--qa_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--qw_qkv_mode', type=str, default='sawb', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--qa_qkv_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--bmm1_qm1_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--bmm1_qm2_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--bmm2_qm1_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--bmm2_qm2_mode', type=str, default='pact', help='activation quantization, pick from lpuq, lsq or qil') 
parser.add_argument('--pact_a_lr', default=0.01, type=float, help='clip val learning rate') 
parser.add_argument('--pact_w_lr', default=0.01, type=float, help='clip val learning rate') 
parser.add_argument('--a_clip_val', type=float, default=6.0, help='clip_val initial value')
parser.add_argument('--a_clip_valn', type=float, default=0.0, help='clip_valn initial value, specifically for QIL')
parser.add_argument('--w_clip_val', type=float, default=1.0, help='positive weight clip_val initial value')   
parser.add_argument('--w_clip_valn', type=float, default=-1.0, help='negative weight clip_val initial value')
parser.add_argument('--pact_a_decay', default=5e-5, type=float, help='clip val for qil pruning clip decay') 
parser.add_argument('--pact_w_decay', default=5e-5, type=float, help='clip val for W decay') 
parser.add_argument('--align_zero',  action='store_true', help='set align_zero flags in W and A quantizers to True')
parser.add_argument('--sentient_check',  action='store_true')
parser.add_argument('--Qmodel_calibration',  default=0, type=int, help='Num of batches for Qmodel calibration')
parser.add_argument('--Qmodel_calibration_new',  default=0, type=int, help='new method for calibration')
parser.add_argument('--QKVsync',  action='store_true', help='synchronize clipvals of QKV layers')
parser.add_argument('--clip_val_asst_percentile', nargs='+', type=float, default=(0.1,99.9), help='pecentile for clip_val initialization')
parser.add_argument('--dropout_prob_attn', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
parser.add_argument('--dropout_prob_hid', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
parser.add_argument('--dropout_prob_emb', type=float, default=0.1, help='in hf3 we changed all dropout prob to 0.165')
parser.add_argument('--plotSVG',  action='store_true', help='save computation graphs, needs graphviz/pygraphviz')

def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print('        - {} : {}'.format(k, v))
        print('=' * 100)


def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    # return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)
    return tuple(
        tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
        for layer_past in past
    )


def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor, 
    num_hypos: int, 
    no_repeat_ngram_size: int, 
    cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs, 
    batch_size, 
    num_beams, 
    prev_output_tokens, 
    repetition_penalty
):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)

        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,                                
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
                history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(
    best_score, 
    best_sequence, 
    batch_size, 
    num_beams, 
    beam_scores, 
    history, 
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** args.length_penalty

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


def beam(model, data_iter, args, sqcfg):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    with torch.no_grad():
        with patch_torch_bmm(sqcfg):
            for idx, data in enumerate(data_iter):
                data = {key: value for key, value in data.items()}

                _id = data['id'].to(args.device)
                _query = data['query'].to(args.device)
                _query_len = data['query_len'].to(args.device)
                _query = _query[:,:_query_len] # only load valid tokens, to work with HF GPT2 without past_len parameter


                ## local adaptation start.

                ## local adaptation end.


                output = None
                score = None

                batch_size = _id.size(0)
                num_beams = args.beam
                length_penalty = args.length_penalty

                _batch = torch.arange(0, _id.size(0), device=args.device, dtype=torch.long)
                
                past = None
                len_past = None

                _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
                _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

                _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)
                _attention_mask = (_query > 0).int()
                
                # scores for each sentence in the beam
                beam_scores = torch.zeros(
                    (batch_size, num_beams), dtype=torch.float, device=_query.device
                )

                best_sequence = torch.zeros(
                    (batch_size, args.eval_len), dtype=torch.long, device=_query.device
                )
                best_score = {}

                history = None
                with torch.no_grad():
                    for i in range(0, args.eval_len):
                        if i == 0:
                            logits, past = model(_query) 
                            logits = logits[_bbatch, (_query_len-1).long(), :] # batch_size * beam, vocab
                        else:
                            #print('token_id.shape', token_id.shape, token_id)
                            #print('past.shape', past[0].shape)
                            #print('len_past.shape', len_past.shape, len_past)
                            
                            position_ids = (len_past).unsqueeze(1)
                            logits, past = model(token_id, past=past, len_past=len_past, position_ids=position_ids) 

                            logits = logits[:, -1, :]    # batch_size * beam, vocab

                        logits = _postprocess_next_token_scores(           
                            logits,
                            history,
                            i,
                            batch_size,
                            num_beams,
                            repetition_penalty=args.repetition_penalty,                                
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            min_length=args.min_length,
                            eos_token_id=args.eos_token_id,
                        )

                        softmax_probs = F.softmax(logits, dim=-1)
                        ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                        vocab_size = softmax_probs.shape[-1] 
                        

                        _logprob = torch.log(softmax_probs) # batch_size * beam, vocab
                        if i == 0:
                            next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab
                            
                        else:
                            next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                            next_scores = next_scores.view(batch_size, -1) # batch_size, beam * vocab

                        next_scores, next_tokens = torch.topk(
                            next_scores, num_beams, dim=1, largest=True, sorted=True
                        )     # batch_size, num_beams
                        
                        beam_id = (next_tokens // vocab_size).view(-1)    # batch_size * num_beams
                        token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1) # batch_size, num_beams

                        beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                        past = _reorder_cache(past, beam_idx.view(-1))                
                        beam_scores = next_scores # batch_size, num_beams
                        len_past = (_query_len + i).long()

                        if history is None:
                            history = token_id.detach()
                        else:
                            history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()

                        _add_beam_candidate(
                            best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                            eos_token_id=args.eos_token_id
                        )
                    
                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history
                    )


                # with torch.no_grad():
                #     _id = distributed_gather(args, _id)
                #     output = distributed_gather(args, best_sequence)
                #     #score = distributed_gather(args, score)
                #     # distributed_sync(args)

                # For single GPU case:
                output = best_sequence

                if args.rank == 0:
                    _id = _id.view(-1).cpu()
                    output = output.view(-1, output.shape[-1]).cpu()
                    #score = score.view(-1, score.shape[-1]).cpu()

                    for _b in range(0, _id.shape[-1]):
                        _i = int(_id[_b].item())
                        all_predictions[_i] = {}
                        all_predictions[_i]['id'] = _i
                        all_predictions[_i]['predict'] = output[_b].tolist()
                        #all_predictions[_i]['score'] = score[_b].tolist()

                    if idx % 10 == 0:
                        print('inference samples', idx)

    if args.rank == 0:
        pred_file = os.path.join(args.work_dir, args.output_file) 
        print('saving prediction file', pred_file)
        with open(pred_file, 'w') as writer:
            for _i in all_predictions:
                writer.write(json.dumps(all_predictions[_i]) + '\n')
    

if __name__ == '__main__':
    args = parser.parse_args()
    # parse_gpu(args)
    print_args(args)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    valid_data = FT_Dataset(
        args.data, args.batch_size, args.seq_len, args.eval_len, 
    )    
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, num_workers=0, shuffle=False, 
        pin_memory=False, drop_last=False)

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )

    lm_net = GPT2LMModel(config)

    if args.lora_dim > 0:
        # target_modules = ['encoder.*query', 'encoder.*key', 'encoder.*value', 'encoder.*dense']
        # target_modules = ['attn', 'mlp']
        target_modules = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj', 'decoder']
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=args.lora_dim,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout,
                                 target_modules=target_modules)
        lm_net= get_peft_model(lm_net, peft_config)

        lm_net.print_trainable_parameters()
        print(lm_net)

    # ----- added for sq1e -----
    sqcfg = senqnn_config_init(args) # we added/parsed our args in sq_args
    sqcfg['dropout_prob_lora'] = args.lora_dropout
    sqcfg['mapping'][peft_Linear]={"from": peft_Linear, "to": QLoRALinear}

    # prepare the model for quantization
    Qmodel_prep(lm_net, valid_loader, sqcfg, 
                save_fname=''.join((args.work_dir, '/model', '.hf4')))

    print(f"trainable parameters:")
    for name, param in lm_net.named_parameters():
        if param.requires_grad :
            print(f"{name}")
    print(f"frozen parameters:")
    for name, param in lm_net.named_parameters():
        if not param.requires_grad :
            print(f"{name}")

    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        cp = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        # lm_net.load_weight(cp)    
        lm_net.load_state_dict(cp['model_state_dict'])
    lm_net = lm_net.cuda()

    print('model sampling ...')
    beam(lm_net, valid_loader, args, sqcfg)
    # distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)