"""Advanced neural network models for denoising quantum measurement signals."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
        """
        return x + self.pe[:, : x.size(1)]


class TransformerDenoiser(nn.Module):
    """Transformer-based model for quantum signal denoising."""

    def __init__(
        self,
        input_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial projection and reshape
        self.input_proj = nn.Conv1d(input_channels, d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Conv1d(d_model, input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_length]
        Returns:
            Denoised signal of same shape as input
        """
        # Project and reshape for transformer
        x = self.input_proj(x)  # [batch, d_model, seq_length]
        x = x.transpose(1, 2)  # [batch, seq_length, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Project back to original shape
        x = x.transpose(1, 2)  # [batch, d_model, seq_length]
        x = self.output_proj(x)  # [batch, channels, seq_length]

        return x


class WaveletBlock(nn.Module):
    """Wavelet-inspired convolutional block."""

    def __init__(self, channels: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, k, padding=k // 2, padding_mode="replicate"
                )
                for k in kernel_sizes
            ]
        )
        self.fusion = nn.Conv1d(channels * len(kernel_sizes), channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale convolutions
        outs = [conv(x) for conv in self.convs]
        # Concatenate along channel dimension
        out = torch.cat(outs, dim=1)
        # Fuse different scales
        out = self.fusion(out)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class WaveletUNet(nn.Module):
    """U-Net architecture with wavelet-inspired blocks."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_sizes: list = [3, 5, 7],
    ):
        super().__init__()

        # Initial projection
        self.input_conv = nn.Conv1d(
            input_channels, hidden_channels, 7, padding=3, padding_mode="replicate"
        )

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = hidden_channels
        for _ in range(num_blocks):
            self.encoder.append(WaveletBlock(ch, kernel_sizes))
            self.downsample.append(nn.Conv1d(ch, ch * 2, 2, stride=2))
            ch *= 2

        # Middle
        self.middle = WaveletBlock(ch, kernel_sizes)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for _ in range(num_blocks):
            self.upsample.append(nn.ConvTranspose1d(ch, ch // 2, 2, stride=2))
            ch //= 2
            self.decoder.append(WaveletBlock(ch, kernel_sizes))

        # Output projection
        self.output_conv = nn.Conv1d(
            hidden_channels, input_channels, 7, padding=3, padding_mode="replicate"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x = self.input_conv(x)

        # Encoder
        encoder_features = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_features.append(x)
            x = down(x)

        # Middle
        x = self.middle(x)

        # Decoder
        for dec, up, skip in zip(
            self.decoder, self.upsample, reversed(encoder_features)
        ):
            x = up(x)
            # Handle odd sequence lengths
            if x.size(-1) != skip.size(-1):
                x = torch.nn.functional.pad(x, (0, skip.size(-1) - x.size(-1)))
            x = x + skip  # Skip connection
            x = dec(x)

        # Output projection
        x = self.output_conv(x)

        return x


class HybridDenoiser(nn.Module):
    """Hybrid model combining Transformer and WaveletUNet approaches."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        num_wavelet_blocks: int = 3,
    ):
        super().__init__()

        # Wavelet U-Net for initial denoising
        self.wavelet_unet = WaveletUNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_wavelet_blocks,
        )

        # Transformer for refining transitions
        self.transformer = TransformerDenoiser(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(input_channels * 2, hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, input_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WaveletUNet path
        wavelet_out = self.wavelet_unet(x)

        # Transformer path
        transformer_out = self.transformer(x)

        # Combine outputs
        combined = torch.cat([wavelet_out, transformer_out], dim=1)
        out = self.fusion(combined)

        return out


if __name__ == "__main__":
    # Test models
    batch_size = 4
    seq_length = 1000
    input_channels = 2

    x = torch.randn(batch_size, input_channels, seq_length)

    # Test Transformer
    print("Testing TransformerDenoiser...")
    transformer = TransformerDenoiser(input_channels)
    y_transformer = transformer(x)
    print(f"Transformer output shape: {y_transformer.shape}")

    # Test WaveletUNet
    print("\nTesting WaveletUNet...")
    wavelet_unet = WaveletUNet(input_channels)
    y_wavelet = wavelet_unet(x)
    print(f"WaveletUNet output shape: {y_wavelet.shape}")

    # Test Hybrid
    print("\nTesting HybridDenoiser...")
    hybrid = HybridDenoiser(input_channels)
    y_hybrid = hybrid(x)
    print(f"Hybrid output shape: {y_hybrid.shape}")

    # Verify shapes
    assert x.shape == y_transformer.shape == y_wavelet.shape == y_hybrid.shape
    print("\nAll tests passed!")
