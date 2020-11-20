import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from code.modules.encoder.highway import Highway
from code.modules.conv1d import Initialized_Conv1d
from code.modules.encoder.depthwise_conv import DepthwiseSeparableConv

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, w_emb_size, c_emb_size, p_drop):
        super(Embedding, self).__init__()
        self.p_drop = p_drop
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors, freeze = True)
        self.conv2d = DepthwiseSeparableConv(c_emb_size, c_emb_size, 5, dim=2)
        self.resize = Initialized_Conv1d(w_emb_size + c_emb_size, hidden_size, bias=False)
        self.hwy = Highway(2, hidden_size, p_drop)

    def forward(self, wx, cx):
        wd_emb = self.w_embed(wx)   # (batch_size, seq_len, word_embed_size)
        ch_emb = self.c_embed(cx)   # (batch_size, seq_len, max_char = 16, char_embed_size)
        ch_emb = ch_emb.permute(0, 3, 1, 2) 
        ch_emb = F.dropout(ch_emb, p=self.p_drop/2, training=self.training) # in the paper it is 0.05, = 0.1/2
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3) # max pooling
        ch_emb = ch_emb.squeeze()

        wd_emb = F.dropout(wd_emb, p=self.p_drop, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([wd_emb, ch_emb], dim=1)
        emb = self.resize(emb)  # (batch_size, hidden_size, seq_len)
        emb = self.hwy(emb)   # (batch_size, hidden_size, seq_len)

        return emb.transpose(1,2)

