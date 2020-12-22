import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dim, bias=True):
        super(DepthwiseSeparableConv, self).__init__()

        # see https://arxiv.org/pdf/1706.03059.pdf for depth and pointwise convolution
        # see https://stackoverflow.com/questions/44212831/convolutional-nn-for-text-input-in-pytorch for doubts
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=kernel_size, groups=in_ch, padding=kernel_size // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=kernel_size, groups=in_ch, padding=kernel_size // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        return self.pointwise_conv(self.depthwise_conv(x))
