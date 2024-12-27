"""Inference utilities for quantum signal denoising models."""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict

from denoiser.model import DenoisingCNN
from denoiser.model_v3 import UNET
from denoiser.config import (
    CNNConfig,
    UNETConfig,
    DEVICE,
    MEAS_TIME,
    DEFAULT_MODEL,
)
from denoiser.curriculum_stages import generate_curriculum_data, get_curriculum_stages


def load_model_config(run_dir: Path) -> Dict:
    """Load model configuration from a training run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def get_model_instance(model_name: str) -> torch.nn.Module:
    """Get model instance based on architecture name."""
    if model_name == "cnn_v1":
        return DenoisingCNN(
            input_channels=CNNConfig.INPUT_CHANNELS,
            hidden_channels=CNNConfig.HIDDEN_CHANNELS,
            num_residual_blocks=CNNConfig.NUM_RESIDUAL_BLOCKS,
            kernel_size=CNNConfig.KERNEL_SIZE,
        )
    elif model_name == "unet":
        return UNET(
            UNETConfig.INPUT_CHANNELS, UNETConfig.OUT_CHANNELS, UNETConfig.FEATURES
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def load_model(
    run_dir: Union[str, Path], device: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a trained model and its configuration.

    Args:
        run_dir: Path to the training run directory containing model checkpoint and config
        device: Device to load model on. If None, uses config.DEVICE

    Returns:
        Tuple of (loaded_model, model_config)
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load configuration
    config = load_model_config(run_dir)
    model_name = config.get("model_name", DEFAULT_MODEL)

    # Initialize model
    model = get_model_instance(model_name)

    # Load weights
    checkpoint_path = run_dir / "checkpoints" / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = device or DEVICE
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, config


def denoise_signal(
    signal: np.ndarray,
    run_dir: Union[str, Path],
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Denoise quantum measurement signals using a trained model.

    Args:
        signal: Complex-valued numpy array of shape (sequence_length,) for single instance
               or (batch_size, sequence_length) for batch
        run_dir: Path to the training run directory containing model and config
        device: Device to run inference on. If None, uses config.DEVICE

    Returns:
        Denoised signal as complex-valued numpy array of same shape as input
    """
    # Load model
    model, _ = load_model(run_dir, device)
    device = device or DEVICE

    # Validate input
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.dtype != np.complex128 and signal.dtype != np.complex64:
        raise TypeError("Signal must be complex-valued")

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
    return denoised[0] if single_instance else denoised


def plot_denoising_result(
    noisy_signal: np.ndarray,
    denoised_signal: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot comparison of noisy and denoised signals."""
    import matplotlib.pyplot as plt

    time_axis = time_axis if time_axis is not None else MEAS_TIME
    time_us = time_axis * 1e6  # Convert to microseconds

    plt.figure(figsize=(12, 5))

    # Plot I component
    plt.subplot(1, 2, 1)
    plt.plot(time_us, noisy_signal.real, "gray", alpha=0.5, label="Noisy")
    plt.plot(time_us, denoised_signal.real, "r", label="Denoised")
    plt.title("I Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot Q component
    plt.subplot(1, 2, 2)
    plt.plot(time_us, noisy_signal.imag, "gray", alpha=0.5, label="Noisy")
    plt.plot(time_us, denoised_signal.imag, "r", label="Denoised")
    plt.title("Q Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage with curriculum data generation
    import matplotlib.pyplot as plt

    # Get latest model from results
    results_dir = Path("results")
    latest_run = max(results_dir.glob("*"), key=lambda p: p.stat().st_mtime)

    # Generate example data using curriculum stages
    stage = get_curriculum_stages()[0]  # Use first stage parameters
    noisy_data, clean_data = generate_curriculum_data(stage)

    # Select a single example
    example_idx = np.random.randint(len(noisy_data))
    noisy_signal = noisy_data[example_idx]
    clean_signal = clean_data[example_idx]

    # Denoise signal
    denoised_signal = denoise_signal(noisy_signal, latest_run)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot I component
    plt.subplot(1, 3, 1)
    plt.plot(MEAS_TIME * 1e6, noisy_signal.real, "gray", alpha=0.5, label="Noisy")
    plt.plot(MEAS_TIME * 1e6, clean_signal.real, "g", label="Clean")
    plt.plot(MEAS_TIME * 1e6, denoised_signal.real, "r--", label="Denoised")
    plt.title("I Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot Q component
    plt.subplot(1, 3, 2)
    plt.plot(MEAS_TIME * 1e6, noisy_signal.imag, "gray", alpha=0.5, label="Noisy")
    plt.plot(MEAS_TIME * 1e6, clean_signal.imag, "g", label="Clean")
    plt.plot(MEAS_TIME * 1e6, denoised_signal.imag, "r--", label="Denoised")
    plt.title("Q Component")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot complex plane
    plt.subplot(1, 3, 3)
    plt.plot(noisy_signal.real, noisy_signal.imag, "gray", alpha=0.5, label="Noisy")
    plt.plot(clean_signal.real, clean_signal.imag, "g", label="Clean")
    plt.plot(denoised_signal.real, denoised_signal.imag, "r--", label="Denoised")
    plt.title("IQ Plane")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
