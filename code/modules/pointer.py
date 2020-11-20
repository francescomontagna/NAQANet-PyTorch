import torch
import torch.nn as nn

from code.modules.utils import mask_logits


class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W1 = nn.Linear(d_model*2, 1)
        self.W2 = nn.Linear(d_model*2, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, M:list, c_mask:torch.tensor):
        M1, M2, M3 = M[0], M[1], M[2]
        X1 = torch.cat([M1, M2], dim=-1)
        X2 = torch.cat([M1, M3], dim=-1)
        Y1 = self.W1(X1).squeeze(-1)
        Y2 = self.W2(X2).squeeze(-1)
        Y1 = mask_logits(Y1, c_mask)
        Y2 = mask_logits(Y2, c_mask)
        span_start_index = self.softmax(Y1)
        span_end_index = self.softmax(Y2)
        return span_start_index, span_end_index