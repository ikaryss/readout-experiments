# engine.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from datagen import DataGenerator


# ============================================================
# 1. Prepare Train / Validation Dataloaders
# ============================================================
def prepare_dataloaders(
    batch_size=32, train_ratio=0.8, num_samples=1000, signal_length=100, seed=None
):
    """
    Prepare training and validation dataloaders using the DataGenerator.

    Args:
        batch_size (int): Batch size for training/validation.
        train_ratio (float): Fraction of data used for training.
        num_samples (int): Total number of samples to generate.
        signal_length (int): Length of time series.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
    """
    generator = DataGenerator(signal_length=signal_length, seed=seed)

    # Generate full dataset
    X_jump, (y_amp_jump, y_cp_jump) = generator.generate_data(
        num_samples // 2, jump=True
    )
    X_const, (y_amp_const, y_cp_const) = generator.generate_data(
        num_samples // 2, jump=False
    )

    # Concatenate jump and constant datasets
    X = torch.cat([X_jump, X_const], dim=0)
    y_amp = torch.cat([y_amp_jump, y_amp_const], dim=0)
    y_cp = torch.cat([y_cp_jump, y_cp_const], dim=0)

    # Shuffle dataset
    indices = torch.randperm(num_samples)
    X, y_amp, y_cp = X[indices], y_amp[indices], y_cp[indices]

    # Train/validation split
    train_size = int(train_ratio * num_samples)
    val_size = num_samples - train_size

    X_train, y_amp_train, y_cp_train = (
        X[:train_size],
        y_amp[:train_size],
        y_cp[:train_size],
    )
    X_val, y_amp_val, y_cp_val = X[train_size:], y_amp[train_size:], y_cp[train_size:]

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_amp_train, y_cp_train)
    val_dataset = TensorDataset(X_val, y_amp_val, y_cp_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ============================================================
# 2. Training and Validation Functions
# ============================================================
def train_epoch(model, dataloader, optimizer, device, amp_loss_fn, cp_loss_fn):
    """
    Run one training epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training DataLoader.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Device to run on (CPU/GPU).
        amp_loss_fn (Loss): Loss function for amplitude regression.
        cp_loss_fn (Loss): Loss function for change point localization.

    Returns:
        avg_loss (float): Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for X, y_amp, y_cp in dataloader:
        X, y_amp, y_cp = X.to(device), y_amp.to(device), y_cp.to(device)

        optimizer.zero_grad()
        amp_pred, cp_pred, _ = model(X)

        loss_amp = amp_loss_fn(amp_pred, y_amp)
        loss_cp = cp_loss_fn(cp_pred, y_cp)
        loss = loss_amp + loss_cp

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate(model, dataloader, device, amp_loss_fn, cp_loss_fn):
    """
    Run validation.

    Args:
        model (nn.Module): Model to validate.
        dataloader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run on (CPU/GPU).
        amp_loss_fn (Loss): Loss function for amplitude regression.
        cp_loss_fn (Loss): Loss function for change point localization.

    Returns:
        avg_loss (float): Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y_amp, y_cp in dataloader:
            X, y_amp, y_cp = X.to(device), y_amp.to(device), y_cp.to(device)

            amp_pred, cp_pred, _ = model(X)

            loss_amp = amp_loss_fn(amp_pred, y_amp)
            loss_cp = cp_loss_fn(cp_pred, y_cp)
            loss = loss_amp + loss_cp

            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


# ============================================================
# 3. Model Training Loop
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    amp_loss_fn,
    cp_loss_fn,
    num_epochs=20,
    plot_loss=True,
):
    """
    Train the model and track performance metrics.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on (CPU/GPU).
        amp_loss_fn (Loss): Loss function for amplitude regression.
        cp_loss_fn (Loss): Loss function for change point localization.
        num_epochs (int): Number of training epochs.
        plot_loss (bool): Whether to plot loss curves.

    Returns:
        trained_model (nn.Module): Best trained model.
    """
    best_val_loss = float("inf")
    best_model_wts = model.state_dict()
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, amp_loss_fn, cp_loss_fn
        )
        val_loss = validate(model, val_loader, device, amp_loss_fn, cp_loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    print("Training complete. Best Val Loss: {:.4f}".format(best_val_loss))

    if plot_loss:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
        plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()

    return model


# ============================================================
# Test Script (Run this section to test engine)
# ============================================================
if __name__ == "__main__":
    from model import ChangePointDetectionModel  # Import model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = prepare_dataloaders(num_samples=1000, seed=42)

    model = ChangePointDetectionModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    amp_loss_fn = nn.MSELoss()
    cp_loss_fn = nn.MSELoss()

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        amp_loss_fn,
        cp_loss_fn,
        num_epochs=20,
    )
