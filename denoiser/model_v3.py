"""U-Net model for IQ signal denoising."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]
    ) -> None:
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose1d(feature * 2, feature, 2, 2))


def test_model():
    """Test model with random input."""
    model = UNET()
    x = torch.randn(1, 2, 500)  # [batch_size, channels, sequence_length]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_model()
