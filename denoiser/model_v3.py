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
    def __init__(self, in_channels=2, out_channels=2, features=None):
        super(UNET, self).__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(
                    feature * 2, feature, kernel_size=2, stride=2, bias=False
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.output = nn.Conv1d(features[0], out_channels, 1)

    def forward(self, x):
        "forward pass of x: (batch, ch, len)"
        residual_x_list = []

        # Down part of UNET
        for down in self.downs:
            x = down(x)
            residual_x_list.append(x)
            x = self.pool(x)

        # Bottleneck part of UNET
        x = self.bottleneck(x)

        # reverse residual inputs
        residual_x_list = residual_x_list[::-1]

        # Up part of UNET
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat((residual_x_list[i // 2], x), dim=1)
            x = self.ups[i + 1](x)

        # Output part of UNET
        return self.output(x)


def test_model():
    """Test model with random input."""
    model = UNET()
    x = torch.randn(1, 2, 512)  # [batch_size, channels, sequence_length]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_model()
