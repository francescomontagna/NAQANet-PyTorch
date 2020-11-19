from modules.encoder.encoder import (EncoderBlock, Highway)
from modules.pointer import Pointer
from modules.cq import CQAttention
from modules.embeddings import Embedding

import torch
import torch.nn as nn

from modules.utils import set_mask, get_embeddings
from .util import torch_from_json
from .args import get_train_args

# TODO
# Check Mask still workin
# BIDAF attention layer?


class QANet(nn.Module):
    def __init__(self, 
                 device,
                 word_embeddings,
                 char_embeddings,
                 w_emb_size:int = 300,
                 c_emb_size:int = 64,
                 hidden_size:int = 128,
                 c_max_len: int = 800,
                 q_max_len: int = 100,
                 p_dropout: float = 0.1,
                 num_heads : int = 8): # need info for padding?
        """
        :param hidden_size: hidden size of representation vectors
        :param q_max_len: max number of words in a question sentence
        :param c_max_len: max number of words in a context sentence
        :param p_dropout: dropout probability
        """
        super(QANet, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.dropout_layer = torch.nn.Dropout(p=p_dropout) if p_dropout > 0 else lambda x: x

        self.embeddings = Embedding(word_embeddings, char_embeddings, hidden_size, w_emb_size, c_emb_size, p_dropout)

        self.context_encoder = EncoderBlock(device, hidden_size, c_max_len, num_convs=4, kernel_size=7, p_dropout=p_dropout, num_heads=num_heads)
        self.question_encoder = EncoderBlock(device, hidden_size, q_max_len, num_convs=2, kernel_size=5, p_dropout=p_dropout,  num_heads=num_heads)

        self.cq_attention = CQAttention(hidden_size, p_dropout)

        self.modeling_resizing_layer = nn.Linear(4 * hidden_size, hidden_size)
        self.modeling_encoder_layer = EncoderBlock(device, hidden_size, len_sentence=c_max_len, p_dropout=0.1)

        self.pointer = Pointer(hidden_size) # forward method return start and end spans


    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):

        cb = self.embeddings(cw_idxs, cc_idxs)
        qb = self.embeddings(qw_idxs, qc_idxs)

        # masks for self attention
        c_mask_enc = set_mask(cb, negated=False).to(self.device)
        q_mask_enc = set_mask(qb, negated=False).to(self.device)

        # masks for CQ attention
        c_mask_c2q = set_mask(cb, negated=True).to(self.device) # ~c_mask_enc
        q_mask_c2q = set_mask(qb, negated=True).to(self.device) # ~q_mask_enc

        cb, qb = self.context_encoder(cb, c_mask_enc), self.question_encoder(qb, q_mask_enc)

        X = self.cq_attention(cb, qb, c_mask_c2q, q_mask_c2q)

        modeled_passage_list = [self.modeling_resizing_layer(X)]
        for _ in range(3):
            modeled_passage = self.dropout_layer(
                self.modeling_encoder_layer(modeled_passage_list[-1], c_mask_enc)
            )
            modeled_passage_list.append(modeled_passage)
        # Pop the first one, which is input
        modeled_passage_list.pop(0)

        span_start, span_end = self.pointer(modeled_passage_list, c_mask_c2q)

        # best_spans = get_best_span(span_start, span_end)
        # print(best_spans)

        return span_start, span_end


if __name__ == "__main__":
    test = False
    torch.emp

    if test:

        torch.manual_seed(32)
        batch_size = 4
        q_max_len = 5
        c_max_len = 7
        wemb_vocab_size = 9
        emb_size = 300
        hidden_size = 128
        dropout_prob = 0.1
        device = 'cpu'
        args = get_train_args()
        word_embeddings = torch_from_json(args.word_emb_file)

        # define model
        model = QANet(device, word_embeddings)

        # fake dataset
        question_lengths = torch.LongTensor(batch_size).random_(1, q_max_len)
        question_wids = torch.zeros(batch_size, q_max_len).long()
        context_lengths = torch.LongTensor(batch_size).random_(1, c_max_len)
        context_wids = torch.zeros(batch_size, c_max_len).long()

        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = \
                torch.LongTensor(1, question_lengths[i]).random_(
                    1, wemb_vocab_size)

            context_wids[i, 0:context_lengths[i]] = \
                torch.LongTensor(1, context_lengths[i]).random_(
                    1, wemb_vocab_size)

        p1, p2 = model(context_wids, question_wids)

        yp1 = torch.argmax(p1, 1)
        yp2 = torch.argmax(p2, 1)
        yps = torch.stack([yp1, yp2], dim=1)
        print(f"yp1 {yp1}")
        print(f"yp2 {yp2}")
        print(f"yps {yps}")

        ymin, _ = torch.min(yps, 1)
        ymax, _ = torch.max(yps, 1)
        print(f"ymin {ymin}")
        print(f"ymax {ymax}")

