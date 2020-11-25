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
from code.util import collate_fn, SQuAD, convert_tokens, discretize
from code.args import get_train_args
from code.model.naqanet import NAQANet

# TODO In realta devo agire sul testo, oerche mancano molti numeri nei word_emb
def get_number_idxs(word2idx_file: str):
    """
    :param word2idx_path: path of word2idxs.json file, mapping word into vocbulary indices
    """
    
    with open(word2idx_file, 'r') as file:
        data = file.read()
    word2idx = json.loads(data)
    
    number_indices = [] # idx associated to numbers
    regex = re.compile('[+-]?[0-9]+\.[0-9]+') # for float
    for key, value in word2idx.items():
        if key.isdigit() or regex.search(key):
            number_indices.append(value)
        else:
            try:
                words2num(key)
                number_indices.append(value)
            except:
                pass

    return number_indices


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

    # Set list with number indices in the vocabulary. Sorted
    number_emb_idxs = get_number_idxs(args.word2idx_file)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    # Get model
    log.info('Building model...')
    model = NAQANet(device, word_vectors, char_vectors)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    
    model = nn.DataParallel(model, args.gpu_ids)
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

    # set loss
    criterion = nn.NLLLoss(reduction = 'mean') # LogSoftmax applied in Pointer

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
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
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Find indices of numbers in the contexts. Shape: (# True values, 2)
                # start_time = time.time()
                number_indices = np.argwhere((np.isin(cw_idxs.numpy(),number_emb_idxs)))
                # print("--- %s seconds ---" % (time.time() - start_time))

                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, number_indices)
                y1, y2 = y1.to(device), y2.to(device)
                loss = criterion(log_p1, y1) + criterion(log_p2, y2)
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
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())