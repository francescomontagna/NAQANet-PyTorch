import torch
import torch.nn as nn
import torch.nn.functional as F

# 2 layer Highway network, nothing to add
# For deeper knowledge: https://arxiv.org/pdf/1505.00387.pdf and
#                       https://arxiv.org/pdf/1611.01603.pdf



class Highway(nn.Module):
    """
    Applies highway transformation to the incoming data.
    It is like LSTM that uses gates. Highway network is
    helpful to train very deep neural networks.
    y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)
    C = 1 - T
    :Examples:
        >>> m = Highway(2, 300)
        >>> x = torch.randn(32, 20, 300)
        >>> y = m(x)
        >>> print(y.size())
    """

    def __init__(self, layer_num, size):
        """
        :param layer_num: number of highway transform layers
        :param size: size of the last dimension of input = embedding size
        """
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(self.n)])
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :Input: (N, *, size) * means any number of additional dimensions.
        :Output: (N, *, size) * means any number of additional dimensions.
        """
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

if __name__ == "__main__":

    highway_network = Highway(2, 300)
    x = torch.randn((2, 4, 300))
    print(highway_network(x) - x)