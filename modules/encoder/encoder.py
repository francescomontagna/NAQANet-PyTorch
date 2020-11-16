import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader # The embedding retrieval is already done in dataset __getitem__ method
import os
import pandas as pd
from sample.bangliu import EncoderBlock as EB
from modules.encoder.depthwise_conv import DepthwiseSeparableConv
from modules.encoder.highway import Highway
from modules.encoder.residual_with_layer_dropout import ResidualWithLayerDropout
from sample.sample_vocab import SampleVocab
from sample.sample_dataset import MultilingualDataset, generate_batch
from modules.utils import set_mask

# Note: see https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py for pre processing

class PositionalEncoding(nn.Module):  # is this model wrapped in recurrent structure?
    def __init__(self, device, d_model, max_len=300):
        """
        :param d_model: dimension of the embedding (after 1st convolution, that bring emb_size from 300 to 128)
        :param p_dropout: dropout probability
        :param max_len: max length of the sentence sequence

        Defines the positional encoding to be summed to input embedding
        """
        super(PositionalEncoding, self).__init__()

        # Computed once for all. See "Attention is all you need for reference"
        self.pos_encoding = torch.zeros((max_len, d_model))  # max_len x d_model matrix
        position = torch.arange(0, max_len)  # variable part of the encoding

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pos_encoding[:, 0::2] = torch.sin(torch.transpose(position.unsqueeze(0), 0, 1) * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(torch.transpose(position.unsqueeze(0), 0, 1) * div_term)

        self.pos_encoding = self.pos_encoding.unsqueeze(0).to(device)

    def forward(self, x):
        """

        :param x: tensor of size batch_size x d_model ( sure? )
        :return: input tensor encoding positional information
        """

        return x + self.pos_encoding[:, :x.size(1)]  # why requires_grad = False?


# https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/ ottima spiegazione attention
# Only for debugging
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(SelfAttention, self).__init__()
        self.self_attention_layer = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x):
        return self.self_attention_layer(x, x, x, need_weights = True) # per capire x,x,x : https://www.reddit.com/r/pytorch/comments/c2u6g5/pytorch_multihead_attention_module/



class EncoderBlock(nn.Module):
    def __init__(self, device, d_model:int, len_sentence: int,  num_convs = 4, kernel_size = 7, p_dropout = 0.1, num_heads = 8):
        super(EncoderBlock, self).__init__()

        self.d_model = d_model  # size of embeddings of a word
        self.len_sentence = len_sentence
        self.dropout = nn.Dropout(p_dropout)
        self.residual_p_dropout = 0.1

        self.pos_enc_layer = PositionalEncoding(device, d_model,
                                                len_sentence) # once for every block (verify)

        self.conv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_convs)])
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(num_convs):
            padding = torch.nn.ConstantPad1d(
                (kernel_size // 2, (kernel_size - 1) // 2), 0
            )
            depthwise_conv = torch.nn.Conv1d(
                d_model, d_model, kernel_size, groups=d_model
            )
            pointwise_conv = torch.nn.Conv1d(d_model, d_model, 1)
            activation_layer = nn.ReLU()
            self.conv_layers.append(
                torch.nn.Sequential(
                    padding, depthwise_conv, pointwise_conv, activation_layer
                )
            )
        self.residual_with_layer_dropout = ResidualWithLayerDropout(True, self.residual_p_dropout)

        self.self_attention_norm = nn.LayerNorm(d_model) # before self attention
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.0) # Probably remove p_dropout

        self.ff_norm = nn.LayerNorm(d_model)  # before FFN
        self.ff_layer = nn.Linear(d_model, d_model)



    def forward(self, x, mask):

        output = self.pos_enc_layer(x)

        # Counter for stochastic dropout within encoder layers
        total_sublayers = len(self.conv_layers) + 2
        sublayer_count = 0

        # Dropout: https://github.com/allenai/allennlp-models/blob/master/allennlp_models/rc/modules/seq2seq_encoders/qanet_encoder.py
        for conv_norm_layer, conv_layer in zip(self.conv_norms, self.conv_layers):
            conv_norm_out = self.dropout(conv_norm_layer(output))
            conv_out = self.dropout(conv_layer(conv_norm_out.transpose_(1, 2)).transpose_(1, 2))
            sublayer_count += 1

            # Resiudal connection + stochastic depth layer dropout
            output = self.residual_with_layer_dropout(
                output, conv_out, sublayer_count, total_sublayers
            )

        norm_out = self.dropout(self.self_attention_norm(output)).transpose(1, 0)
        attention_out = self.dropout(self.self_attention(norm_out, norm_out, norm_out, need_weights = False, key_padding_mask=mask)[0]) # mask? See AllenNLP
        attention_scores = self.dropout(self.self_attention(norm_out, norm_out, norm_out, need_weights = True, key_padding_mask = mask)[1])
        sublayer_count += 1
        output = self.residual_with_layer_dropout(
            output, attention_out.transpose(1,0), sublayer_count, total_sublayers
        )

        feedforward_norm_out = self.dropout(self.ff_norm(output))
        feedforward_out = self.dropout(self.ff_layer(feedforward_norm_out))
        sublayer_count += 1
        output = self.residual_with_layer_dropout(
            output, feedforward_out, sublayer_count, total_sublayers
        )
        return output


if __name__ == "__main__":
    torch.manual_seed(13)
    test_PosEncoder = False
    test_DepthConv = False
    test_SelfAtt = False
    test_EncoderBlock = True

    # dataset
    batch_size = 2
    working_dir = os.getcwd()
    vocab = SampleVocab(os.path.join(working_dir, os.pardir, "sample_embeddings.txt"))
    data = pd.read_csv(os.path.join(working_dir, os.pardir, "sample_dataset.txt"), encoding="utf-8")

    dataset1 = DataLoader(MultilingualDataset(data.text, data.labels, vocab, torch.int32),
                         shuffle=True,
                         batch_size=batch_size,
                         collate_fn=generate_batch
                         )

    dataset2 = DataLoader(MultilingualDataset(data.text, data.labels, vocab, torch.int32),
                         shuffle=True,
                         batch_size=batch_size,
                         collate_fn=generate_batch
                         )

    dataset = dataset1

    if test_PosEncoder:
        pe = PositionalEncoding(300, 500)
        for x, y in dataset:

            print(x) # UNDERSTAND PADDING
            # print(pe(x))

    if test_DepthConv:
        highway = Highway(2, 300)
        conv_layer = DepthwiseSeparableConv(300, 128, 7)
        pe = PositionalEncoding(300, 500)
        for x, _ in dataset:
            print(x)
            x = highway(x)
            x = pe(x)
            # print(f"Wrong features size {x.size()}")
            # print(f"Correct features size {torch.transpose(x, 1, 2).size()}")
            output = conv_layer(torch.transpose(x, 1, 2))
            # print(f"Output features size {output.size()}") # transposition due to depthwise approach. Correct, done also by reference github repo
            break

    if test_SelfAtt:
        pe = PositionalEncoding(300, 500)

        resizing_proj_layer = torch.nn.Linear(300, 128)
        sa = SelfAttention(128, 8, dropout=0) # 8 from the paper
        for x, _ in dataset:
            x = pe(x)
            x = resizing_proj_layer(x)

            # query size :
            # is ( N, L, E)  where L = sequence length, N = batch size, E = embedding size
            # should be (L, N, E)
            res = x
            print(f"Wrong features size: {x.size()}")
            x = x.transpose(1,0)

            print(f"Correct features size before attention: {x.size()}") # check if it is the expected size
            x, attention_scores = sa(x)
            print("Embedding size after attention: {}".format(x.size()))
            print("Residual size: {}".format(res.size()))
            print("In order to sum attention and residual, need to transpose again x.transpose(1,0)")
            attention_out = res + x.transpose(1,0)

            print(attention_out)
            print(attention_out.size())

            # print("Attention scores size: {}".format(attention_scores.size())) # size = (N, L, L) ==> per ogni batch, per ogni parola nella frase, ho la sua attenzione rispetto alle altre parole

            break

    if test_EncoderBlock:
        emb_size = 300
        d_model = 128

        resizing_projection_layer = torch.nn.Linear(emb_size,
                                                         d_model)  # code says Linear, paper says 1D conv TODO check
        highway = Highway(2, d_model)
        encoder = EncoderBlock(d_model, 500)

        for x, _ in dataset1:
            c_mask = set_mask(x)
            # print(c_mask)

            x = highway(resizing_projection_layer(x))
            y = encoder(x, c_mask)
            print(f"Mio: {y}")
            break

    if test_EncoderBlock:
        batch_size = 2
        seq_length = 4
        hidden_dim = 128
        resizing_projection_layer = torch.nn.Linear(300, 128)
        for x, _ in dataset2:
            m = EB(4, 128, 8, 7, dropout=0.1)
            y = m(resizing_projection_layer(x).transpose(1, 2), mask=c_mask)
            print(f"Bangliu: {y.transpose(1,2)}")
            break