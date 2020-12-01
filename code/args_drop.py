"""Command-line arguments for setup.py, train.py, test.py.
Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse
import os

cwd = os.getcwd()
data_path = os.path.join(cwd, "data", "drop")


def get_setup_drop_args():
    """Get arguments needed in setup.py."""
    print(f"CWD: {cwd}")
    parser = argparse.ArgumentParser('Download and pre-process SQuAD')

    add_common_args(parser)

    parser.add_argument('--train_file',
                        type=str,
                        default=os.path.join(data_path , 'drop_dataset_train.json'))
    parser.add_argument('--dev_file',
                        type=str,
                        default=os.path.join(data_path , 'drop_dataset_dev.json'))
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--glove_path',
                        type=str,
                        default=os.path.join(cwd + '/data/glove.840B.300d/glove.840B.300d.txt'))
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default=os.path.join(data_path, 'dev_meta.json'))
    parser.add_argument('--word2idx_file',
                        type=str,
                        default=os.path.join(data_path, 'word2idx.json'))
    parser.add_argument('--char2idx_file',
                        type=str,
                        default=os.path.join(data_path, 'char2idx.json'))
    parser.add_argument('--answer_file',
                        type=str,
                        default=os.path.join(data_path, 'answer.json'))
    parser.add_argument('--para_limit',
                        type=int,
                        default=1926,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=70,
                        help='Max number of words to keep from a question')
    parser.add_argument('--num_idx_limit',
                        type=int,
                        default=90,
                        help='Max of \'number indices\' in a context')
    parser.add_argument('--spans_limit',
                        type=int,
                        default=44,
                        help='Max answer spans in a context')
    parser.add_argument('--counts_limit',
                        type=int,
                        default=1,
                        help='Max number of numerical answers in a context')
    parser.add_argument('--as_expr_limit',
                        type=int,
                        default=403,
                        help='Max number of addition/subtraction signs for an answer')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')
    parser.add_argument('--include_test_examples',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Process examples from the test set')

    args = parser.parse_args()

    return args


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

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
        '-g',
        '--gpu_ids',
        type = int,
        action='append',
        help = 'gpu ids')


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
        help = 'dropout probability between layers')
    parser.add_argument('--eval_steps',
        type=int,
        default=87561,
        help='Number of steps between successive evaluations.')
    
    # metrics & checkpoints
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    # seed
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


# def get_test_args():
#     """Get arguments needed in test.py."""
#     parser = argparse.ArgumentParser('Test a trained model on SQuAD')

#     add_common_args(parser)
#     add_train_test_args(parser)

#     parser.add_argument('--split',
#                         type=str,
#                         default='dev',
#                         choices=('train', 'dev', 'test'),
#                         help='Split to use for testing.')
#     parser.add_argument('--sub_file',
#                         type=str,
#                         default='submission.csv',
#                         help='Name for submission file.')

#     # Require load_path for test.py
#     args = parser.parse_args()
#     if not args.load_path:
#         raise argparse.ArgumentError('Missing required argument --load_path')

#     return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--train_record_file',
                        type=str,
                        default=os.path.join(data_path, 'train.npz'))
    parser.add_argument('--dev_record_file',
                        type=str,
                        default=os.path.join(data_path,'dev.npz'))
    parser.add_argument('--word_emb_file',
                        type=str,
                        default=os.path.join(data_path, 'word_emb.json'))
    parser.add_argument('--char_emb_file',
                        type=str,
                        default=os.path.join(data_path, 'char_emb.json'))
    parser.add_argument('--train_eval_file',
                        type=str,
                        default=os.path.join(data_path, 'train_eval.json'))
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default=os.path.join(data_path, 'dev_eval.json'))


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        default = 'train', type=str,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=30,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')