"""Inference utilities for the denoising model."""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional

from denoiser.model import DenoisingCNN
from denoiser.config import (
    INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    NUM_RESIDUAL_BLOCKS,
    KERNEL_SIZE,
    DEVICE,
)


def denoise_signal(
    signal: np.ndarray, model_path: Union[str, Path], device: Optional[str] = None
) -> np.ndarray:
    """
    Denoise quantum measurement signals using a trained model.

    Args:
        signal: Complex-valued numpy array of shape (sequence_length,) for single instance
               or (batch_size, sequence_length) for batch
        model_path: Path to the saved model weights (.pt file)
        device: Device to run inference on. If None, uses config.DEVICE

    Returns:
        Denoised signal as complex-valued numpy array of same shape as input
    """
    # Ensure model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Set device
    if device is None:
        device = DEVICE

    # Initialize model
    model = DenoisingCNN(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        kernel_size=KERNEL_SIZE,
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Handle single instance vs batch
    single_instance = signal.ndim == 1
    if single_instance:
        signal = signal[np.newaxis, :]

    # Convert complex data to I/Q channels
    channels = np.stack((signal.real, signal.imag), axis=1)

    # Convert to tensor
    x = torch.FloatTensor(channels).to(device)

    # Perform inference
    with torch.no_grad():
        y = model(x)

    # Convert back to numpy and combine channels
    y_np = y.cpu().numpy()
    denoised = y_np[:, 0] + 1j * y_np[:, 1]

    # Return single instance or batch based on input
    if single_instance:
        return denoised[0]
    return denoised


if __name__ == "__main__":
    # Example usage
    from data_generator import generate_relaxation_data
    import matplotlib.pyplot as plt

    # Generate example noisy data
    meas_time = np.arange(0, 2e-6, 2e-9)
    noisy_data, _ = generate_relaxation_data(
        batch_size=1,
        meas_time=meas_time,
        I_amp_1=-200,
        Q_amp_1=400,
        I_amp_0=-250,
        Q_amp_0=200,
        relax_time_transition=1e-9,
        T1=50e-6,
        gauss_noise_amp=1300,
        qubit=1,
        seed=42,
    )

    # Get latest model from results
    results_dir = Path("results")
    latest_run = max(results_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    model_path = latest_run / "checkpoints" / "model.pt"

    # Denoise signal
    denoised_data = denoise_signal(noisy_data[0], model_path)

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot I component
    plt.subplot(1, 2, 1)
    plt.plot(meas_time * 1e6, noisy_data[0].real, "gray", alpha=0.5, label="Noisy")
    plt.plot(meas_time * 1e6, denoised_data.real, "r", label="Denoised")
    plt.title("I Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot Q component
    plt.subplot(1, 2, 2)
    plt.plot(meas_time * 1e6, noisy_data[0].imag, "gray", alpha=0.5, label="Noisy")
    plt.plot(meas_time * 1e6, denoised_data.imag, "r", label="Denoised")
    plt.title("Q Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
