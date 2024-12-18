"""Configuration parameters for the denoising model."""

import numpy as np

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("PyTorch not found. Please install it with: pip install torch")
    DEVICE = "cpu"

# Data Generation
DATA_GEN_BATCH_SIZE = 80000
PER_CLASS_BATCH_SIZE = 3000  # Large enough for stable training
NOISE_AMP = 1300
MEAS_TIME = np.arange(0, 2e-6, 2e-9)
SEQUENCE_LENGTH = len(MEAS_TIME)  # 1000 points

# Model Architecture
INPUT_CHANNELS = 2  # I and Q components
HIDDEN_CHANNELS = 64
NUM_RESIDUAL_BLOCKS = 4
KERNEL_SIZE = 5

# Training
TRAIN_VAL_SPLIT = 0.8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64

# Random Seeds
RANDOM_SEED = 42
DATA_SEED = 42
