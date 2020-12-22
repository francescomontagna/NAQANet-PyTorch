import torch
import torch.nn as nn
import torch.nn.functional as F

from code.modules.conv1d import Initialized_Conv1d


class Highway(nn.Module):
    def __init__(self, layer_num: int, size, dropout):
        super().__init__()
        self.n = layer_num
        self.dropout = dropout
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            #x = F.relu(x)
        return x