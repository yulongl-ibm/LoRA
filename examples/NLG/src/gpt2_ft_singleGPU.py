#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import torch
import random
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora.layer import Linear as peft_Linear

from sq1e.sen_qnn_ext import Qmodel_prep, senqnn_config_init, patch_torch_bmm
from sq1e.sen_qnn_infer import QLoRALinear
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')
# add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

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

# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args,
    sqcfg,
    train_step=0, 
    epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    # train_loader.sampler.set_epoch(epoch)

    with patch_torch_bmm(sqcfg):
        for idx, data in enumerate(train_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            # _lm_logits, _lm_loss = model(
            #     _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
            # ) 
            _lm_logits, _lm_loss = model(
                input_ids=_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
            ) 

            _lm_loss = _lm_loss.mean() 

            train_step += 1
            is_update = True if train_step % args.grad_acc == 0 else False
            avg_lm_loss.update(_lm_loss.item())
            optimizer_step(
                _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
            )
            
            if train_step % args.log_interval == 0: 
                elapsed = time.time() - log_start_time
                lr = optimizer.param_groups[0]['lr']
                log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                        f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                        f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                        f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

                for k, v in model.named_parameters():
                    if 'clip_val' in k: tb_writer.add_scalar(k, v, train_step)
                tb_writer.add_scalar("training loss", avg_lm_loss.val, train_step)
                tb_writer.add_scalar("avg training loss", avg_lm_loss.avg, train_step)
                tb_writer.add_scalar("ppl", math.exp(avg_lm_loss.avg), train_step)

                if args.rank == 0: 
                    print(log_str)
                log_start_time = time.time()
                avg_lm_loss.reset()
            
            if train_step % args.save_interval == 0: 
                if args.rank == 0:
                    model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                    print('saving checkpoint', model_path)
                    # torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
                # distributed_sync(args)

            # evaluation interval
            if train_step % args.eval_interval == 0:
                eval_start_time = time.time()

                valid_loss, valid_ppl = evaluate(model, valid_loader, args)

                if best_val_ppl is None or valid_ppl < best_val_ppl:
                    best_val_ppl = valid_ppl
                    
                log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                        f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                        f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

                tb_writer.add_scalar("valid loss", valid_loss, train_step)
                tb_writer.add_scalar("valid_ppl", valid_ppl, train_step)

                if args.rank == 0:
                    print('-' * 100)
                    print(log_str)
                    print('-' * 100)

                model.train()
                # distributed_sync(args)

            if train_step == args.max_step:
                break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    # distributed_sync(args)
    return train_step


if __name__ == '__main__':
    args = parser.parse_args()
    # parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        # sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        # sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            # lora_attn_dim=args.lora_dim, 
            # lora_attn_alpha=args.lora_alpha, 
            # lora_dropout=args.lora_dropout,
            # --- adjust dropout prob ---
            # attention_probs_dropout_prob=args.dropout_prob_attn,
            # hidden_dropout_prob=args.dropout_prob_hid,
            # embedding_dropout_prob=args.dropout_prob_emb,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))    

    lm_net = lm_net.cuda()

    # if args.lora_dim > 0:
    #     lora.mark_only_lora_as_trainable(lm_net)

    # settings for LoRA
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

    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    # lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    # ----- added for sq1e -----
    sqcfg = senqnn_config_init(args) # we added/parsed our args in sq_args
    tb_writer=SummaryWriter(log_dir=f"{args.work_dir}/runs")
    sqcfg['dropout_prob_lora'] = args.lora_dropout
    sqcfg['mapping'][peft_Linear]={"from": peft_Linear, "to": QLoRALinear}

    # prepare the model for quantization
    if sqcfg['Qmodel_calibration'] > 0:
        Qmodel_prep(lm_net, train_loader, sqcfg, 
                    optimizer=optimizer, scheduler=scheduler,
                    prefwdproc=lambda datamb: (datamb['input_ids'].to(args.device),), 
                    save_fname=''.join((args.work_dir, '/model', '.hf4')))
    else:
        Qmodel_prep(lm_net, train_loader, sqcfg, 
                    optimizer=optimizer, scheduler=scheduler,
                    save_fname=''.join((args.work_dir, '/model', '.hf4')))

    print(f"trainable parameters:")
    for name, param in lm_net.named_parameters():
        if param.requires_grad :
            print(f"{name}")
    print(f"frozen parameters:")
    for name, param in lm_net.named_parameters():
        if not param.requires_grad :
            print(f"{name}")

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, sqcfg,
                train_step=train_step, epoch=epoch
            )
            
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    # distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)
