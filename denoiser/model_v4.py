import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation_rate=1):
        """
        A single residual block with two convolutional layers.
        The dilation_rate allows us to adjust the receptive field without increasing the kernel size.
        """
        super(ResidualBlock, self).__init__()
        # Calculate padding to keep the output length same as input length.
        padding = ((kernel_size - 1) // 2) * dilation_rate

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation_rate
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation_rate
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class Denoising1DModel(nn.Module):
    def __init__(
        self, input_channels=2, num_filters=64, num_residual_blocks=6, kernel_size=3
    ):
        """
        Denoising model for 1D signals with two channels (in-phase and quadrature).
        The architecture includes:
          - An initial convolution for feature extraction.
          - A series of residual blocks (with optionally alternating dilation rates).
          - A final convolution to reconstruct the clean signal.
        """
        super(Denoising1DModel, self).__init__()

        # Initial feature extraction convolution
        self.initial_conv = nn.Conv1d(
            input_channels, num_filters, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.relu = nn.ReLU(inplace=True)

        # Create a sequence of residual blocks.
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            # Alternate dilation rates (e.g., 2 for even-indexed blocks, 1 for odd-indexed)
            dilation_rate = 2 if i % 2 == 0 else 1
            self.residual_blocks.append(
                ResidualBlock(num_filters, kernel_size, dilation_rate=dilation_rate)
            )

        # Final reconstruction convolution mapping back to the 2-channel output.
        self.final_conv = nn.Conv1d(
            num_filters, input_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        """
        Forward pass.
        Input x is expected to have shape: (batch_size, length, channels).
        Since PyTorch's Conv1d expects (batch_size, channels, length), we permute dimensions.
        """
        out = self.initial_conv(x)
        out = self.relu(out)
        for block in self.residual_blocks:
            out = block(out)
        out = self.final_conv(out)
        return out


# Test the model with a random input tensor
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, length, channels)
    batch_size = 8
    signal_length = 1024
    num_channels = 2
    model = Denoising1DModel(
        input_channels=num_channels,
        num_filters=64,
        num_residual_blocks=6,
        kernel_size=3,
    )

    print(model)

    sample_input = torch.randn(batch_size, num_channels, signal_length)
    output = model(sample_input)
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
