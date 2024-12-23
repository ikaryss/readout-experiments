"""Training script for advanced denoising models."""

# uv run .\denoiser\train_v2.py --model hybrid
# pylint: disable=unused-wildcard-import, wildcard-import

import argparse
import json
import sys
from datetime import datetime
from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from denoiser.config import *
from denoiser.curriculum_stages import generate_curriculum_data, get_curriculum_stages
from denoiser.engine import prepare_data, train_model
from denoiser.model import DenoisingCNN  # Original model
from denoiser.model_v3 import UNET

# Add parent directory to path to import data_generator
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)


def get_model(model_name: str) -> nn.Module:
    """
    Get model instance based on name.

    Args:
        model_name: Name of the model to instantiate

    Returns:
        Instantiated model
    """
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
        raise ValueError(f"Unknown model: {model_name}")


def setup_directories(model_name: str) -> Path:
    """Create directories for saving results."""
    base_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{model_name}_{timestamp}"

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    return run_dir


def plot_training_history(history: dict, save_dir: Path):
    """Plot and save training metrics."""
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "plots" / "loss.png")
    plt.close()

    # Plot SNR
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_snr"])
    plt.xlabel("Epoch")
    plt.ylabel("SNR (dB)")
    plt.title("Validation SNR")
    plt.grid(True)
    plt.savefig(save_dir / "plots" / "snr.png")
    plt.close()


def plot_example_results(
    model: nn.Module,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    save_dir: Path,
    device: str,
    num_examples: int = 3,
):
    """Plot example denoising results."""
    model.eval()

    # Convert first few examples to tensors
    noisy_channels = np.stack((noisy_data.real, noisy_data.imag), axis=1)
    clean_channels = np.stack((clean_data.real, clean_data.imag), axis=1)

    noisy_tensor = torch.FloatTensor(noisy_channels[:num_examples]).to(device)

    # Generate predictions
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).cpu()

    # Plot results
    _, axes = plt.subplots(num_examples, 2, figsize=(15, 5 * num_examples))

    for i in range(num_examples):
        # Plot I component
        axes[i, 0].plot(
            MEAS_TIME * 1e6, noisy_channels[i, 0], "gray", alpha=0.5, label="Noisy"
        )
        axes[i, 0].plot(MEAS_TIME * 1e6, clean_channels[i, 0], "g", label="Clean")
        axes[i, 0].plot(MEAS_TIME * 1e6, denoised_tensor[i, 0], "r--", label="Denoised")
        axes[i, 0].set_title(f"Example {i+1} - I Component")
        axes[i, 0].set_xlabel("Time (μs)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)
        axes[i, 0].legend()

        # Plot Q component
        axes[i, 1].plot(
            MEAS_TIME * 1e6, noisy_channels[i, 1], "gray", alpha=0.5, label="Noisy"
        )
        axes[i, 1].plot(MEAS_TIME * 1e6, clean_channels[i, 1], "g", label="Clean")
        axes[i, 1].plot(MEAS_TIME * 1e6, denoised_tensor[i, 1], "r--", label="Denoised")
        axes[i, 1].set_title(f"Example {i+1} - Q Component")
        axes[i, 1].set_xlabel("Time (μs)")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].grid(True)
        axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "example_results.png")
    plt.close()


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train denoising model")
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help="Model architecture to use",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    print(f"Utilized device: {DEVICE}")
    print(f"Training model: {args.model}")

    # Setup directories
    run_dir = setup_directories(args.model)

    # Generate dataset
    print("Generating dataset...")
    stages = get_curriculum_stages()

    # Initialize model
    print("Initializing model...")
    model = get_model(args.model).to(DEVICE)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for stage in stages:
        print(stage)
        noisy_data, clean_data = generate_curriculum_data(stage)

        # Prepare data loaders
        print("Preparing data loaders...")
        train_loader, val_loader = prepare_data(
            noisy_data=noisy_data,
            clean_data=clean_data,
            train_ratio=TRAIN_VAL_SPLIT,
            batch_size=TRAIN_BATCH_SIZE,
            device=DEVICE,
        )

        # Train model
        print("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            early_stopping_patience=10,
        )

    # Save results
    print("Saving results...")

    # Save model
    torch.save(model.state_dict(), run_dir / "checkpoints" / "model.pt")

    # Save model configuration
    config = {
        "model_name": args.model,
        "model_config": {
            name: getattr(model, name) for name, param in model.named_parameters()
        },
        "training_config": {
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "early_stopping_patience": 10,
        },
    }

    with open(run_dir / "config.json", "w", encoding="UTF-8") as f:
        json.dump(config, f, indent=4, default=str)

    # Save training history
    with open(run_dir / "history.json", "w", encoding="UTF-8") as f:
        json.dump(history, f, indent=4)

    # Plot results
    plot_training_history(history, run_dir)
    plot_example_results(model, noisy_data, clean_data, run_dir, DEVICE)

    print(f"Training complete! Results saved in {run_dir}")


if __name__ == "__main__":
    main()
