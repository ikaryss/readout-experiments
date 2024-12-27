"""Training and validation engine for the denoising model."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset

from denoiser.metrics import evaluate_batch


def prepare_data(
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare DataLoaders for training and validation.

    Args:
        noisy_data: Complex-valued noisy data array (batch_size, sequence_length)
        clean_data: Complex-valued clean data array (batch_size, sequence_length)
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the training data
        device: Device to load data to

    Returns:
        Tuple of (train_loader, val_loader)
    """

    # Convert complex data to real channels (I/Q)
    def complex_to_channels(data: np.ndarray) -> np.ndarray:
        return np.stack((data.real, data.imag), axis=1)

    # Prepare channel data
    noisy_channels = complex_to_channels(noisy_data)
    clean_channels = complex_to_channels(clean_data)

    # Convert to tensors
    noisy_tensor = torch.FloatTensor(noisy_channels)
    clean_tensor = torch.FloatTensor(clean_channels)

    # Split into train and validation sets
    n_train = int(len(noisy_data) * train_ratio)

    train_noisy = noisy_tensor[:n_train].to(device)
    train_clean = clean_tensor[:n_train].to(device)
    val_noisy = noisy_tensor[n_train:].to(device)
    val_clean = clean_tensor[n_train:].to(device)

    # Create datasets
    train_dataset = TensorDataset(train_noisy, train_clean)
    val_dataset = TensorDataset(val_noisy, val_clean)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)

        # Forward pass
        denoised = model(noisy)
        loss = criterion(denoised, clean)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_snr = 0.0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Forward pass
            denoised = model(noisy)
            loss = criterion(denoised, clean)

            # Calculate metrics
            mse, mae, snr = evaluate_batch(denoised, clean)

            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            total_snr += snr.item()

    # Calculate averages
    num_batches = len(val_loader)
    metrics = {
        "val_loss": total_loss / num_batches,
        "val_mse": total_mse / num_batches,
        "val_mae": total_mae / num_batches,
        "val_snr": total_snr / num_batches,
    }

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: Optional[int] = None,
) -> Dict[str, list]:
    """
    Train the model with validation and optional early stopping.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        early_stopping_patience: Number of epochs to wait for improvement
            before early stopping. If None, no early stopping is used.

    Returns:
        Dictionary containing training history
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_snr": [],
        "epochs": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        metrics = validate(model, val_loader, criterion, device)

        # Update history
        history["train_loss"].append(train_loss)
        for key, value in metrics.items():
            history[key].append(value)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {metrics['val_loss']:.6f}")
        print(f"Val SNR: {metrics['val_snr']:.2f} dB")
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["val_loss"])
            else:
                scheduler.step()

        print(f"Learning Rate: {current_lr:.2e}")
        print("-" * 40)

        # Early stopping
        if early_stopping_patience is not None:
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    history["epochs"].append(epoch + 1)
    return history
