"""
the progressive residual fusion dense network proposed by Tiantian, Wang ; Hu, Zhihua ; Guan, Yurong
"An efficient lightweight network for image denoising using progressive residual and convolutional attention feature fusion"
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels, out_channels, kernel_size=3, bias=False, padding=1
            ),
        )

    def forward(self, x):
        return self.bottleneck(x)


class DenseBlock(nn.Module):
    """
    Dense Block for 1D signals
    Reference: https://paperswithcode.com/method/dense-block
    """

    def __init__(
        self, in_channels, num_bottlenecks, growth_rate=12, bottleneck_size=48
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_bottlenecks):
            self.layers.append(Bottleneck(in_channels, bottleneck_size, growth_rate))
            in_channels += growth_rate  # Update in_channels for the next layer

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)  # Concatenate along the channel dimension
        return x


# Example Usage
if __name__ == "__main__":
    # Parameters
    in_channels = 16
    num_bottlenecks = 3
    growth_rate = 12
    bottleneck_size = 48

    # Dense Block
    dense_block = DenseBlock(in_channels, num_bottlenecks, growth_rate, bottleneck_size)

    # Test Input
    x = torch.randn(4, in_channels, 100)  # Batch size = 4, Channels = 16, Length = 100
    output = dense_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
