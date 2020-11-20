import argparse
import math
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.process import parse_data, load_json
from data.datasets import MultilingualDataset, generate_batch
from data.vocab import Vocab
from trainer.util import make_eval_dict_tokens
from modules.ema import EMA
from model.qanet import QANet
from trainer.trainer import Trainer
from trainer.util import max_context_question_len

cwd = os.getcwd() # current working directory
parser = argparse.ArgumentParser()

# vocab
parser.add_argument(
    '--language',
    default="en", type=str, # check if en or english or whatever
    help='word embeddings language')
parser.add_argument(
    '--common_embeddings_filepath',
    default='/mnt/data1/nlp/embeddings/fasttext/fasttext.common.vec', #TODO fare attenzione porco dio
    type=str, help='path of common embedding file')
parser.add_argument(
    '--word_embeddings_filepath',
    default='/mnt/data1/nlp/embeddings/fasttext/fasttext.en.vec',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--emb_size',
    default=300, type=int,
    help='word embedding size (default: 300)')

# dataset
parser.add_argument("--version",
                    default = "1.1", type = str,
                    help = "version of the squad dataset: 1.1 or 2.0")
parser.add_argument(
    '--train_data_filepath',
    default=cwd + '/data/squad/1.1/train-v1.1.json',
    type=str, help='path to training dataset file')
parser.add_argument(
    '--dev_data_filepath',
    default=cwd + '/data/squad/1.1/dev-v1.1.json',
    type=str, help='path to dev dataset file')


# train & evaluate
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=30, type=int,
    help='number of total epochs (default: 30)')
parser.add_argument(
    '--p_dropout',
    default = 0.1, type = float,
    help = 'dropout probability between layers'
)
parser.add_argument(
    '--val_num_batches',
    default=500, type=int,
    help='number of batches for evaluation (default: 500)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (default: 2)')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='checkpoints/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='Adam optimizer beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='Adam optimizer beta 2')
parser.add_argument(
    '--decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')

# model
parser.add_argument(
    '--context_limit',
    default=5000, type=int,
    help='maximum context token number')
parser.add_argument(
    '--question_limit',
    default=1000, type=int,
    help='maximum question token number')
parser.add_argument(
    '--answer_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='attention num head')

# cuda
parser.add_argument(
    '--use_gpu',
    default=False, action='store_true',
    help='whether or not train on gpu')
parser.add_argument(
    '--device_id',
    default=-1, type = int,
    help = 'device id for gpu training')

def main(args):
    print("Main is running!")


    squad_path = os.path.join(os.getcwd(),'data',
                                            'squad',
                                            args.version)
    dev_eval_dict_path_from_tokens = os.path.join(squad_path, 'dev_eval_dict_from_tokens.json')
    if 'dev_eval_dict_from_tokens.json' not in os.listdir(squad_path):
        print("Generating valuation dictionary... ", end = "")
        make_eval_dict_tokens(args.dev_data_filepath, dev_eval_dict_path_from_tokens)
        print("Done")

    # set device
    if args.use_gpu:
        device_id = args.device_id
        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")

    # Dataset
    train_json = load_json(args.train_data_filepath)
    eval_json = load_json(args.dev_data_filepath)
    train_data = pd.DataFrame(parse_data(train_json))
    eval_data = pd.DataFrame(parse_data(eval_json))
    header = list(train_data.columns)


    torch.manual_seed(12)
    common_vocab = Vocab(args.language, args.common_embeddings_filepath, args.emb_size)
    vocab = Vocab(args.language, args.word_embeddings_filepath, args.emb_size, base = common_vocab)
    train_dataloader = DataLoader(MultilingualDataset(train_data, vocab),
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  collate_fn=generate_batch)
    val_dataloader = DataLoader(MultilingualDataset(eval_data, vocab),
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  collate_fn=generate_batch)

    # get model
    model = QANet(device, 
                  args.emb_size,
                  args.d_model,
                  args.context_limit,
                  args.question_limit,
                  args.p_dropout)

    # exponential moving average
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    model = model.to(device)

    # optimizer & scheduler
    lr = args.lr
    base_lr = 1.0
    warm_up = args.lr_warm_up_num
    params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(lr=base_lr, betas=(args.beta1, args.beta2), eps=1e-7, weight_decay=3e-7, params=params)
    cr = lr / math.log(warm_up)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log(ee + 1) if ee < warm_up else lr)

    # set loss
    criterion = nn.NLLLoss(reduction = 'mean') # LogSoftmax applied in Pointer

    # checkpoint identifier
    identifier = type(model).__name__ + '_'

    # training and evaluation
    trainer = Trainer(args, device, model, optimizer, scheduler, criterion, train_dataloader,
                      val_dataloader, ema, dev_eval_dict_path_from_tokens, identifier)
    trainer.train()

if __name__ == "__main__":
    main(parser.parse_args())