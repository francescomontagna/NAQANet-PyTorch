"""Train a model on SQuAD.
Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import math
import json
import re
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from collections import OrderedDict
from json import dumps
from words2num import words2num
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

import code.util as util
from code.args_drop import get_train_args
from code.model.naqanet import NAQANet
from code.dataset.drop import collate_fn, DROP
from code.drop_eval.drop_metric import eval_dicts


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    
    # set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_ids[0]))
        args.batch_size *= max(1, len(args.gpu_ids))
        print(f"device is cuda: gpu_ids = {args.gpu_ids}")
    else:
        device = torch.device("cpu")
        print("device is cpu")

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    # Get model
    log.info('Building model...')
    model = NAQANet(device, word_vectors, char_vectors,
        c_max_len = args.context_limit,
        q_max_len = args.question_limit,
        answering_abilities = ['passage_span_extraction', 'counting'],
        max_count = args.max_count) # doesn't large max_count lead to meaningless probability?

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    
    # model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    lr = args.lr
    base_lr = 1.0
    warm_up = args.lr_warm_up_num
    params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(lr=base_lr, betas=(args.beta1, args.beta2), eps=1e-7, weight_decay=3e-7, params=params)
    cr = lr / math.log(warm_up)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log(ee + 1) if ee < warm_up else lr)

    # Get data loader
    log.info('Building dataset...')
    train_dataset = DROP(args.train_record_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = DROP(args.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, \
                    qw_idxs, qc_idxs, \
                    start_idxs, end_idxs, \
                    counts, ids  in train_loader:

                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                start_idxs = start_idxs.to(device)
                end_idxs = end_idxs.to(device)
                counts = counts.to(device)
                ids = ids.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                output_dict = model(cw_idxs, cc_idxs,
                       qw_idxs, qc_idxs, ids,
                       start_idxs, end_idxs, counts)

                loss = output_dict["loss"]
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')


def evaluate(model, data_loader, device, eval_file):
    nll_meter = util.AverageMeter() # ?

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        model.eval_data = gold_dict # pass eval_data as model state
        for cw_idxs, cc_idxs, \
                qw_idxs, qc_idxs, \
                start_idxs, end_idxs, \
                counts, ids   in data_loader:

            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            start_idxs = start_idxs.to(device)
            end_idxs = end_idxs.to(device)
            counts = counts.to(device)
            ids = ids.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            output_dict = model(cw_idxs, cc_idxs,
                       qw_idxs, qc_idxs, ids,
                       start_idxs, end_idxs, counts)
            loss = output_dict['loss']
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            pred_dict.update(output_dict["predictions"])

    model.eval_data = None
    model.train()

    eval_dict = eval_dicts(gold_dict, pred_dict,)
    results_list = [('Loss', nll_meter.avg),
                    ('F1', eval_dict['F1']),
                    ('EM', eval_dict['EM'])]
    
    eval_dict = OrderedDict(results_list)

    return eval_dict, pred_dict


if __name__ == '__main__':
    main(get_train_args())