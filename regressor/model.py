import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Residual Block with Dilated Conv
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        """
        A residual block with two dilated convolutional layers.
        Args:
            channels (int): Number of channels (input and output).
            dilation (int): Dilation rate for the convolutions.
            kernel_size (int): Kernel size (default: 3).
        """
        super(ResidualBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2  # to keep same length
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x  # save for the skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # add skip connection
        out = self.relu(out)
        return out


# ----------------------------
# Amplitude Regression Head
# ----------------------------
class AmplitudeHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, output_dim=4):
        """
        Head for regressing the 4 amplitude values.
        Args:
            in_channels (int): Number of channels coming from the encoder.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of amplitude outputs (4).
        """
        super(AmplitudeHead, self).__init__()
        # Global average pooling will reduce (B, C, T) -> (B, C)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (B, C, T)
        x = torch.mean(x, dim=2)  # global average pooling over time: (B, C)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x  # shape: (B, 4)


# ----------------------------
# Change-Point Localization Head
# ----------------------------
class ChangePointHead(nn.Module):
    def __init__(self, in_channels, length):
        """
        Head for localizing the change point.
        Args:
            in_channels (int): Number of channels from encoder.
            length (int): Length (number of time steps) of the input signal.
        """
        super(ChangePointHead, self).__init__()
        # 1x1 convolution to collapse channels to 1.
        self.conv1x1 = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.length = length

    def forward(self, x):
        # x shape: (B, C, T)
        out = self.conv1x1(x)  # shape: (B, 1, T)
        out = out.squeeze(1)  # shape: (B, T)
        # Compute a probability distribution over time using softmax.
        prob = F.softmax(out, dim=1)  # shape: (B, T)
        # Create a tensor of time indices (0, 1, 2, ..., T-1)
        device = x.device
        indices = torch.arange(
            self.length, dtype=torch.float32, device=device
        )  # shape: (T,)
        indices = indices.unsqueeze(0)  # shape: (1, T) to broadcast over batch
        # Soft-argmax: weighted sum of time indices.
        cp = torch.sum(prob * indices, dim=1)  # shape: (B,)
        return cp, prob  # cp is the predicted change point index (continuous)


# ----------------------------
# Full Model Definition
# ----------------------------
class ChangePointDetectionModel(nn.Module):
    def __init__(self, input_channels=2, num_filters=64, num_blocks=4, length=100):
        """
        A multi-task model that outputs both amplitude regression predictions
        and a change-point localization prediction.
        Args:
            input_channels (int): Number of input channels (2 for in-phase and quadrature).
            num_filters (int): Number of filters (channels) used in the convolutional layers.
            num_blocks (int): Number of residual blocks.
            length (int): Length of the time series.
        """
        super(ChangePointDetectionModel, self).__init__()
        # Initial convolution to expand 2 channels to num_filters.
        self.initial_conv = nn.Conv1d(
            input_channels, num_filters, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks with increasing dilation (e.g., 1, 2, 4, 8).
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(num_filters, dilation=2**i, kernel_size=3)
                for i in range(num_blocks)
            ]
        )

        # Two heads for the two tasks.
        self.amplitude_head = AmplitudeHead(num_filters, hidden_dim=64, output_dim=4)
        self.cp_head = ChangePointHead(num_filters, length=length)

    def forward(self, x):
        # x shape: (B, 2, T)
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)
        for block in self.res_blocks:
            out = block(out)
        # Shared feature maps, shape: (B, num_filters, T)
        amplitude_pred = self.amplitude_head(out)  # (B, 4)
        cp_pred, cp_distribution = self.cp_head(
            out
        )  # cp_pred: (B,), cp_distribution: (B, T)
        return amplitude_pred, cp_pred, cp_distribution


# ----------------------------
# Testing the Model
# ----------------------------
if __name__ == "__main__":
    # Example parameters
    sequence_length = 100  # length of the time series
    batch_size = 8
    input_channels = 2  # in-phase and quadrature channels

    # Instantiate the model
    model = ChangePointDetectionModel(
        input_channels=input_channels,
        num_filters=64,
        num_blocks=4,
        length=sequence_length,
    )

    # Create a dummy input: batch_size x 2 x sequence_length
    dummy_input = torch.randn(batch_size, input_channels, sequence_length)

    # Forward pass
    amplitude_pred, cp_pred, cp_distribution = model(dummy_input)

    print("Amplitude predictions shape:", amplitude_pred.shape)  # Expected: (8, 4)
    print("Change point prediction shape:", cp_pred.shape)  # Expected: (8,)
    print(
        "Change point distribution shape:", cp_distribution.shape
    )  # Expected: (8, 100)

    # Example losses (to be used in training):
    # Suppose we have ground truth amplitudes (shape: batch_size x 4)
    # and ground truth change point indices (normalized to the [0, sequence_length-1] range).
    gt_amplitudes = torch.randn(batch_size, 4)
    gt_cp = torch.randint(0, sequence_length, (batch_size,), dtype=torch.float32)

    # Loss for amplitudes (MSE loss)
    amplitude_loss = F.mse_loss(amplitude_pred, gt_amplitudes)

    # Loss for change point localization (MSE loss)
    cp_loss = F.mse_loss(cp_pred, gt_cp)

    total_loss = amplitude_loss + cp_loss

    print(
        "Amplitude loss: {:.4f}, CP loss: {:.4f}, Total loss: {:.4f}".format(
            amplitude_loss.item(), cp_loss.item(), total_loss.item()
        )
    )
