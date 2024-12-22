"""Custom loss functions for IQ signal denoising."""

import torch
import torch.nn as nn
# import torch.fft as fft


class DenoisingLoss(nn.Module):
    """Compound loss for IQ signal denoising."""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_phase: float = 0.1,
        # lambda_spectral: float = 0.01
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_phase = lambda_phase
        # self.lambda_spectral = lambda_spectral

    def forward(
        self,
        pred: torch.Tensor,  # Shape: [batch_size, 2, sequence_length]
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate compound loss.

        Args:
            pred: Predicted IQ signals [batch_size, 2, sequence_length]
            target: Target IQ signals [batch_size, 2, sequence_length]

        Returns:
            Total weighted loss
        """
        # L1 loss
        l1_loss = torch.mean(torch.abs(pred - target))

        # Convert to complex representation
        pred_complex = pred[:, 0, :] + 1j * pred[:, 1, :]
        target_complex = target[:, 0, :] + 1j * target[:, 1, :]

        # Phase loss
        phase_pred = torch.angle(pred_complex)
        phase_target = torch.angle(target_complex)
        phase_loss = torch.mean(torch.abs(phase_pred - phase_target))

        # Spectral loss
        # pred_fft = fft.fft(pred_complex)
        # target_fft = fft.fft(target_complex)
        # spectral_loss = torch.mean(torch.abs(pred_fft - target_fft))

        # Combine losses
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_phase * phase_loss
            # self.lambda_spectral * spectral_loss
        )

        return total_loss


if __name__ == "__main__":
    # Test loss function
    criterion = DenoisingLoss()

    # Create random test data
    batch_size, sequence_length = 4, 500
    pred = torch.randn(batch_size, 2, sequence_length)
    target = torch.randn(batch_size, 2, sequence_length)

    loss = criterion(pred, target)
    print(f"Test loss value: {loss.item():.6f}")
