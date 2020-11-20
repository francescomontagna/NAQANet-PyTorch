import torch
import  torch.nn as nn
import torch.nn.functional as F
import math
from modules.utils import mask_logits

# Implementa anche il quarto layer! Vedi infatti cosa ritorna :-)
class CQAttention(nn.Module):
    def __init__(self, d_model, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.p_dropout = p_dropout
        w = torch.empty(self.d_model * 3)
        lim = 1 / self.d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        ss = []
        # C = C.transpose(1, 2)
        # Q = Q.transpose(1, 2) # non lo metto perché il mio encoder restituisce le dimensioni corrette
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=self.p_dropout, training=self.training)
        return out