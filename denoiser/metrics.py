"""Metrics for evaluating denoising performance."""

import torch
import numpy as np
from typing import Tuple, Optional
from scipy.signal import find_peaks


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

def calculate_state_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold_excited: float = 0.5
) -> float:
    """
    Calculate state classification accuracy.

    Args:
        pred: Predicted IQ signals [batch_size, 2, sequence_length]
        target: Target IQ signals [batch_size, 2, sequence_length]
        threshold_excited: Threshold for excited state classification

    Returns:
        Classification accuracy
    """
    def classify_state(signal: torch.Tensor) -> torch.Tensor:
        # Convert IQ to amplitude
        amplitude = torch.sqrt(signal[:, 0, :]**2 + signal[:, 1, :]**2)
        # Calculate mean amplitude for each sequence
        mean_amp = amplitude.mean(dim=1)
        # Classify states based on amplitude
        states = torch.where(mean_amp > threshold_excited, 1, 0)
        return states

    pred_states = classify_state(pred)
    target_states = classify_state(target)

    accuracy = (pred_states == target_states).float().mean()
    return accuracy.item()


def calculate_transition_timing(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_height: float = 0.3
) -> Tuple[float, float]:
    """
    Calculate transition timing accuracy for relaxation events.

    Args:
        pred: Predicted IQ signals [batch_size, 2, sequence_length]
        target: Target IQ signals [batch_size, 2, sequence_length]
        min_height: Minimum height for peak detection

    Returns:
        Tuple of (mean timing error, timing error std)
    """
    def find_transition_point(signal: torch.Tensor) -> Optional[int]:
        # Convert to amplitude
        amplitude = torch.sqrt(signal[0, :]**2 + signal[1, :]**2)
        # Calculate derivative
        derivative = torch.diff(amplitude)
        # Find peaks in derivative (transition points)
        peaks, _ = find_peaks(derivative.abs().cpu().numpy(), height=min_height)
        return peaks[0] if len(peaks) > 0 else None

    timing_errors = []

    for i in range(pred.shape[0]):
        pred_point = find_transition_point(pred[i])
        target_point = find_transition_point(target[i])

        if pred_point is not None and target_point is not None:
            timing_errors.append(abs(pred_point - target_point))

    if timing_errors:
        timing_errors = np.array(timing_errors)
        return timing_errors.mean(), timing_errors.std()
    return 0.0, 0.0


def calculate_phase_coherence(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[float, float]:
    """
    Calculate phase coherence metrics.

    Args:
        pred: Predicted IQ signals [batch_size, 2, sequence_length]
        target: Target IQ signals [batch_size, 2, sequence_length]

    Returns:
        Tuple of (mean phase difference, phase difference std)
    """
    # Convert to complex representation
    pred_complex = pred[:, 0, :] + 1j * pred[:, 1, :]
    target_complex = target[:, 0, :] + 1j * target[:, 1, :]

    # Calculate phase difference
    phase_diff = torch.angle(pred_complex) - torch.angle(target_complex)

    # Wrap phase difference to [-π, π]
    phase_diff = torch.remainder(phase_diff + np.pi, 2 * np.pi) - np.pi

    # Calculate statistics
    mean_phase_diff = torch.mean(torch.abs(phase_diff)).item()
    std_phase_diff = torch.std(torch.abs(phase_diff)).item()

    return mean_phase_diff, std_phase_diff


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

    # Test transition timing
    mean_error, std_error = calculate_transition_timing(pred, target)

    # Test phase coherence
    # mean_phase, std_phase = calculate_phase_coherence(pred, target)

    print("Test metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"Transition timing error: {mean_error:.4f} ± {std_error:.4f}")
    # print(f"Phase coherence: {mean_phase:.4f} ± {std_phase:.4f}")
