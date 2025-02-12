import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangePointNet(nn.Module):
    def __init__(self):
        super(ChangePointNet, self).__init__()

        # Shared convolutional layers.
        # Input shape: [batch, channels=2, length=512]
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )

        # Amplitude head: Global pooling and FC for the four amplitude outputs.
        self.amp_pool = nn.AdaptiveAvgPool1d(1)  # collapses the time dimension
        self.fc_amp = nn.Linear(128, 4)  # 4 outputs: start I, start Q, stop I, stop Q

        # Jump head: Preserve time axis to localize the jump.
        # One strategy: collapse channel dimension (by taking the mean) so that we have a feature per time step.
        # Then use a fully connected layer to map the 512–dim vector to a single jump index.
        self.fc_jump = nn.Linear(512, 1)

    def forward(self, x):
        # x shape: [batch, 2, 512]
        x = F.relu(self.conv1(x))  # -> [batch, 16, 512]
        x = F.relu(self.conv2(x))  # -> [batch, 32, 512]
        x = F.relu(self.conv3(x))  # -> [batch, 64, 512]

        # ---- Amplitude branch ----
        # Use global average pooling over the time dimension.
        amp_features = self.amp_pool(x)  # -> [batch, 64, 1]
        amp_features = amp_features.squeeze(-1)  # -> [batch, 64]
        amp_out = self.fc_amp(amp_features)  # -> [batch, 4]

        # ---- Jump branch ----
        # To keep time information, we first collapse the channel dimension:
        # For example, compute the mean over the channels.
        # jump_features = x.mean(dim=1)  # -> [batch, 512]
        # jump_out = self.fc_jump(jump_features)  # -> [batch, 1]
        # jump_out = jump_out.squeeze(-1)  # -> [batch]

        # Concatenate the outputs to get a 5-dimensional output per sample:
        # Order: [start I, start Q, stop I, stop Q, jump index]
        # out = torch.cat([amp_out, jump_out.unsqueeze(1)], dim=1)

        return amp_out


class ShallowRegressionModel(nn.Module):
    def __init__(self):
        super(ShallowRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # Flatten input from (batch_size, 512, 2) to (batch_size, 1024)
        x = x.view(x.size(0), -1)
        x = self.bn1(nn.ReLU()(self.fc1(x)))
        x = self.bn2(nn.ReLU()(self.fc2(x)))
        x = self.fc3(x)
        return x


class ConvAmplitudeEstimator(nn.Module):
    def __init__(self):
        super(ConvAmplitudeEstimator, self).__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Second conv block
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(32)

        # Third conv block
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.bn3 = nn.BatchNorm1d(64)

        # After two pooling layers, the temporal dimension reduces:
        # 1024 -> 512 (after first pooling) -> 256 (after second pooling)
        # Flatten size: channels * length = 64 * 256
        self.fc1 = nn.Linear(64 * 256, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 4)  # Predict 4 amplitude values

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Reduces length to 512
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Reduces length to 256
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HybridAmplitudeEstimator(nn.Module):
    def __init__(self):
        super(HybridAmplitudeEstimator, self).__init__()
        # Input: (batch, 512, 2) -> rearrange to (batch, channels, length) = (batch, 2, 512)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduces length by factor of 2

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.bn3 = nn.BatchNorm1d(64)

        # Additional convolutional block (without dilation)
        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm1d(64)

        # After two pooling operations: 512 -> 256 (after first pooling) -> 128 (after second pooling)
        # Final feature map shape: (batch, 64, 128)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 4)  # 4 amplitude values

    def forward(self, x):
        # Rearrange input: (batch, 512, 2) -> (batch, 2, 512)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # -> (batch, 16, 256)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # -> (batch, 32, 128)
        x = F.relu(self.bn3(self.conv3(x)))  # -> (batch, 64, 128)
        x = F.relu(self.bn4(self.conv4(x)))  # -> (batch, 64, 128)
        x = x.view(x.size(0), -1)  # Flatten -> (batch, 64*128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
