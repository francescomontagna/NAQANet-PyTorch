import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias = True):
        super(DepthwiseSeparableConv, self).__init__()

        # see https://arxiv.org/pdf/1706.03059.pdf for depth and pointwise convolution
        # see https://stackoverflow.com/questions/44212831/convolutional-nn-for-text-input-in-pytorch for doubts
        self.depthwise_conv = nn.Conv1d(in_channels = in_ch, #300 or 128 = embed size. Sarebbe un  solo channel con Conv2d. ma visto che uso Conv1d, diventano molti channels con convolution depthwise
                                         out_channels =in_ch, #300 or 128, to have depthwise
                                         kernel_size=kernel_size, #7, value from paper
                                         groups=in_ch, # to have depthwise
                                         padding=kernel_size // 2,
                                         bias=bias
                                         )
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch,
                                        out_channels=out_ch,
                                        kernel_size=1,
                                        padding=0,
                                        bias=bias)

        # initialization
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        """

        :param x: torch.tensor of size (batch_size, sequence_length, embedding_dimension)
        :return: torch.tensor resulting from convolution over input. New size = (batch_size, sequence_length, 128)
        """

        return self.pointwise_conv(self.depthwise_conv(x))