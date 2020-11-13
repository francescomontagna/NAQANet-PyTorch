import torch.nn as nn
from modules.cq import CQAttention
from modules.encoder.encoder import EncoderBlock
from modules.encoder.highway import Highway
from modules.pointer import Pointer
from modules.utils import *
from data.process import load_json, parse_data
import pandas as pd


"""
bangliu usa 7 (?!) encoder blocks nel layer 4
inoltre il resizing lo fa con un layer convoluzionale
"""


# c'è una corrispondenza 1:1 - context:question. Date N domande per 1 context, il context è ripetuto N volte
def main():
    torch.manual_seed(32)
    batch_size = 1
    q_max_len = 5
    c_max_len = 7
    wemb_vocab_size = 9
    emb_size = 300
    d_model = 128
    dropout_prob = 0.1

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

    context = get_embeddings(context_wids, emb_size)
    questions = get_embeddings(question_wids, emb_size)

    test = True

    # Use Linear layer for resizing, as AllenNLP code
    if test:
        c_mask_enc = set_mask(context,negated = False)
        q_mask_enc = set_mask(questions, negated=False)

        resizing_projection_layer = torch.nn.Linear(emb_size,
                                                    d_model)  # code says Linear, paper says 1D conv TODO check
        highway = Highway(2, d_model)
        context_encoder = EncoderBlock(d_model, len_sentence = c_max_len, num_heads = 8, num_convs = 4, kernel_size = 7, p_dropout = 0.1)
        question_encoder = EncoderBlock(d_model, len_sentence = q_max_len, num_heads = 8, num_convs = 4, kernel_size = 7, p_dropout = 0.1)
        cq_att = CQAttention(d_model, 0.1)
        dropout_layer = torch.nn.Dropout(p=dropout_prob) if dropout_prob > 0 else lambda x: x
        modeling_resizing_layer = nn.Linear(4 * d_model, d_model)
        modeling_encoder_layer = EncoderBlock(d_model, len_sentence=c_max_len, num_heads=8, num_convs=4, kernel_size=7,
                                              p_dropout=0.1)  # Perché c_max_len? Perché cerco indici di inizio e fine nel contesto
        pointer = Pointer(d_model)

        cb, qb = highway(resizing_projection_layer(context)), highway(resizing_projection_layer(questions))
        cb, qb = context_encoder(cb, c_mask_enc), question_encoder(qb, q_mask_enc)

        c_mask_c2q = set_mask(context, negated=True)
        q_mask_c2q = set_mask(questions, negated=True)

        X = cq_att(cb, qb, c_mask_c2q, q_mask_c2q) # X.size() = (batch_size, context_length, d_model*4) ==> need to resize


        modeled_passage_list = [modeling_resizing_layer(X)]
        for _ in range(3):
            modeled_passage = dropout_layer(
                modeling_encoder_layer(modeled_passage_list[-1], c_mask_enc)
            )
            modeled_passage_list.append(modeled_passage)
        # Pop the first one, which is input
        modeled_passage_list.pop(0)

        span_start_index, span_end_index = pointer(modeled_passage_list, c_mask_c2q)

        best_spans = get_best_span(span_start_index, span_end_index)
        print(best_spans)

if __name__ == "__main__":
    # main()
    cwd = os.getcwd()
    train_path = os.path.join(cwd, os.pardir, 'data', 'train-v2.0.json')
    train_json = load_json(train_path)
    train_data = pd.DataFrame(parse_data(train_json))
    header = list(train_data.columns)
    print(f"Data header: {header}")
    print(train_data.label.head()) # index based. Model returns label token based ==> ??




