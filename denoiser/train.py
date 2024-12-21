"""Main script for training the denoising model."""

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
from denoiser.engine import prepare_data, train_model
from denoiser.model import DenoisingCNN

# Add parent directory to path to import data_generator
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)


def setup_directories():
    """Create directories for saving results."""
    base_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    return run_dir


def generate_dataset():
    """Generate balanced dataset with ground, excited, and relaxation states."""
    from data_generator import IQParameters, QuantumStateGenerator, RelaxationParameters

    # Initialize generator
    excited = IQParameters(I_amp=EXCITED_I, Q_amp=EXCITED_Q)
    ground = IQParameters(I_amp=GROUND_I, Q_amp=GROUND_Q)
    relaxation = RelaxationParameters(
        T1=T1_TIME, relax_time_transition=RELAX_TRANSITION_TIME
    )

    generator = QuantumStateGenerator(
        meas_time=MEAS_TIME,
        excited_params=excited,
        ground_params=ground,
        relaxation_params=relaxation,
        gauss_noise_amp=NOISE_AMP,
        qubit=0,
    )

    # Generate noisy data
    ground_data_noisy, ground_labels = generator.generate_ground_state(
        batch_size=BATCH_GROUND, seed=DATA_SEED
    )
    excited_data_noisy, excited_labels = generator.generate_excited_state(
        batch_size=BATCH_EXCITED, seed=DATA_SEED
    )
    relax_data_noisy, relax_labels = generator.generate_relaxation_event(
        batch_size=RELAXATION_BATCH, uniform_sampling=True, seed=DATA_SEED
    )

    # Generate clean data (low noise)
    generator.gauss_noise_amp = 10  # Low noise for clean data
    ground_data_clean, _ = generator.generate_ground_state(
        batch_size=BATCH_GROUND, seed=DATA_SEED
    )
    excited_data_clean, _ = generator.generate_excited_state(
        batch_size=BATCH_EXCITED, seed=DATA_SEED
    )
    relax_data_clean, _ = generator.generate_relaxation_event(
        batch_size=RELAXATION_BATCH, uniform_sampling=True, seed=DATA_SEED
    )

    # Combine data
    noisy_data = np.concatenate(
        [ground_data_noisy, excited_data_noisy, relax_data_noisy]
    )
    clean_data = np.concatenate(
        [ground_data_clean, excited_data_clean, relax_data_clean]
    )

    # Shuffle data
    shuffle_idx = np.random.permutation(len(noisy_data))
    noisy_data = noisy_data[shuffle_idx]
    clean_data = clean_data[shuffle_idx]

    return noisy_data, clean_data


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
    clean_tensor = torch.FloatTensor(clean_channels[:num_examples])

    # Generate predictions
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).cpu()

    # Plot results
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 5 * num_examples))

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
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    print(f"Utilized device: {DEVICE}")

    # Setup directories
    run_dir = setup_directories()

    # Generate dataset
    print("Generating dataset...")
    noisy_data, clean_data = generate_dataset()

    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader = prepare_data(
        noisy_data=noisy_data,
        clean_data=clean_data,
        train_ratio=TRAIN_VAL_SPLIT,
        batch_size=TRAIN_BATCH_SIZE,
        device=DEVICE,
    )

    # Initialize model
    print("Initializing model...")
    model = DenoisingCNN(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # Save training history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Plot results
    plot_training_history(history, run_dir)
    plot_example_results(model, noisy_data, clean_data, run_dir, DEVICE)

    print(f"Training complete! Results saved in {run_dir}")


if __name__ == "__main__":
    main()
