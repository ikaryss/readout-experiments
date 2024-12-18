"""Metrics for evaluating denoising performance."""

import torch
import numpy as np
from typing import Tuple


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error between predicted and target signals.

    Args:
        pred: Predicted (denoised) signal of shape (batch_size, 2, sequence_length)
        target: Target (clean) signal of same shape

    Returns:
        MSE value as a scalar tensor
    """
    return torch.mean((pred - target) ** 2)


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Error between predicted and target signals.

    Args:
        pred: Predicted (denoised) signal of shape (batch_size, 2, sequence_length)
        target: Target (clean) signal of same shape

    Returns:
        MAE value as a scalar tensor
    """
    return torch.mean(torch.abs(pred - target))


def calculate_snr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate Signal-to-Noise Ratio in decibels.

    SNR = 10 * log10(signal_power / noise_power)
    where noise is the difference between prediction and target

    Args:
        pred: Predicted (denoised) signal of shape (batch_size, 2, sequence_length)
        target: Target (clean) signal of same shape

    Returns:
        SNR value in dB as a scalar tensor
    """
    # Calculate noise as the difference between prediction and target
    noise = pred - target

    # Calculate powers (mean squared values)
    signal_power = torch.mean(target**2)
    noise_power = torch.mean(noise**2)

    # Avoid division by zero and log of zero
    eps = 1e-10
    snr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))

    return snr


def evaluate_batch(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate all metrics for a batch of predictions.

    Args:
        pred: Predicted (denoised) signal of shape (batch_size, 2, sequence_length)
        target: Target (clean) signal of same shape

    Returns:
        Tuple of (MSE, MAE, SNR) values
    """
    mse = calculate_mse(pred, target)
    mae = calculate_mae(pred, target)
    snr = calculate_snr(pred, target)

    return mse, mae, snr


if __name__ == "__main__":
    # Test metrics with random data
    batch_size = 4
    sequence_length = 1000
    channels = 2

    # Create random target and slightly noisy prediction
    target = torch.randn(batch_size, channels, sequence_length)
    pred = target + 0.1 * torch.randn_like(target)

    # Calculate metrics
    mse, mae, snr = evaluate_batch(pred, target)

    print(f"Test metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"SNR: {snr:.2f} dB")
