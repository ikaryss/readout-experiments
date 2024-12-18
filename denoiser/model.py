"""Neural network model for denoising quantum measurement signals."""

import torch
import torch.nn as nn
from denoiser.config import (
    INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    NUM_RESIDUAL_BLOCKS,
    KERNEL_SIZE,
    SEQUENCE_LENGTH,
)


class ResidualBlock(nn.Module):
    """Residual block with 1D convolutions."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, padding_mode="replicate"
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, padding_mode="replicate"
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DenoisingCNN(nn.Module):
    """Convolutional neural network for denoising quantum signals."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_residual_blocks: int,
        kernel_size: int,
    ):
        super().__init__()

        # Initial feature extraction
        self.input_conv = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_channels, kernel_size)
                for _ in range(num_residual_blocks)
            ]
        )

        # Output projection
        self.output_conv = nn.Conv1d(
            hidden_channels,
            input_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 2, sequence_length)
               where channel 0 is I component and channel 1 is Q component

        Returns:
            Denoised signal of same shape as input
        """
        # Initial convolution
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = self.relu(out)

        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)

        # Output projection
        out = self.output_conv(out)

        return out


if __name__ == "__main__":
    # Test model with random input
    model = DenoisingCNN(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        kernel_size=KERNEL_SIZE,
    )

    # Create random input (batch_size=4, channels=2, sequence_length=1000)
    x = torch.randn(4, INPUT_CHANNELS, SEQUENCE_LENGTH)

    # Forward pass
    y = model(x)

    # Verify output shape matches input shape
    assert x.shape == y.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"
    print("Model test passed!")
