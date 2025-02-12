import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """1D complex-valued convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv1d, self).__init__()
        self.real_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.imag_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, x):
        # x: (batch_size, 2, seq_len) - 2 channels for I/Q
        real = self.real_conv(x[:, 0:1, :]) - self.imag_conv(x[:, 1:2, :])  # Real part
        imag = self.real_conv(x[:, 1:2, :]) + self.imag_conv(
            x[:, 0:1, :]
        )  # Imaginary part
        return torch.cat((real, imag), dim=1)  # Concatenate along channel dimension


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism."""

    def __init__(self, embed_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)  # (batch_size, seq_len, embed_dim)
        v = self.value(x)  # (batch_size, seq_len, embed_dim)

        attn_weights = F.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1
        )
        attn_output = torch.bmm(attn_weights, v)  # (batch_size, seq_len, embed_dim)
        return attn_output


class ChangePointDetector(nn.Module):
    """Main model for change-point detection and amplitude regression."""

    def __init__(self, seq_len, embed_dim=64):
        super(ChangePointDetector, self).__init__()
        self.seq_len = seq_len

        # Complex-valued convolutional layers
        self.complex_conv1 = ComplexConv1d(1, 64, kernel_size=5, padding=2)
        self.complex_conv2 = ComplexConv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Temporal attention
        self.attention = TemporalAttention(embed_dim=128)

        # Jump localization head
        self.jump_head = nn.Sequential(nn.Conv1d(128, 1, kernel_size=1), nn.Sigmoid())

        # Amplitude regression heads
        self.start_amplitude_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 2)  # Start I/Q amplitudes
        )
        self.stop_amplitude_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 2)  # Stop I/Q amplitudes
        )

        # Jump sample regression head
        self.jump_sample_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)  # Jump sample index
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 2) - I/Q channels
        x = x.permute(0, 2, 1)  # (batch_size, 2, seq_len)

        # Complex convolutions
        x = self.complex_conv1(x)  # (batch_size, 128, seq_len)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 128, seq_len // 2)

        x = self.complex_conv2(x)  # (batch_size, 256, seq_len // 2)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 256, seq_len // 4)

        # Temporal attention
        x = x.permute(0, 2, 1)  # (batch_size, seq_len // 4, 256)
        x = self.attention(x)  # (batch_size, seq_len // 4, 256)

        # Jump localization
        jump_prob = self.jump_head(x.permute(0, 2, 1))  # (batch_size, 1, seq_len // 4)
        jump_prob = jump_prob.squeeze(1)  # (batch_size, seq_len // 4)

        # Amplitude regression
        start_amplitude = self.start_amplitude_head(x[:, 0, :])  # First time step
        stop_amplitude = self.stop_amplitude_head(x[:, -1, :])  # Last time step

        # Jump sample regression
        jump_sample = self.jump_sample_head(x.mean(dim=1))  # Global average pooling

        return {
            "jump_prob": jump_prob,
            "start_amplitude": start_amplitude,
            "stop_amplitude": stop_amplitude,
            "jump_sample": jump_sample,
        }


# Example usage
if __name__ == "__main__":
    seq_len = 100  # Length of input sequence
    batch_size = 16
    model = ChangePointDetector(seq_len=seq_len)

    # Dummy input
    x = torch.randn(batch_size, seq_len, 2)  # (batch_size, seq_len, 2)
    output = model(x)

    print("Jump probability shape:", output["jump_prob"].shape)
    print("Start amplitude shape:", output["start_amplitude"].shape)
    print("Stop amplitude shape:", output["stop_amplitude"].shape)
    print("Jump sample shape:", output["jump_sample"].shape)
