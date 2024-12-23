"""Configuration parameters for the denoising models."""

import numpy as np

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("PyTorch not found. Please install it with: pip install torch")
    DEVICE = "cpu"

# Data Generation
BATCH_GROUND = 3000  # Number of ground state samples
BATCH_EXCITED = 3000  # Number of excited state samples
RELAXATION_BATCH = 3000  # Number of relaxation samples
# MEAS_TIME = np.arange(0, 1.984e-6, 2e-9)
MEAS_TIME = np.arange(0, 1e-6 + 12 * 2e-9, 2e-9)
SEQUENCE_LENGTH = len(MEAS_TIME)
QUBIT = 0

# IQ Parameters
EXCITED_I = [-1000, 1000]
EXCITED_Q = [-1000, 1000]
GROUND_I = [-1000, 1000]
GROUND_Q = [-1000, 1000]
NOISE_AMP = [1000, 2000]
CLEAN_AMP = 10
T1_TIME = 50e-6
RELAX_TRANSITION_TIME = 1e-9

# Model Architectures


# Original CNN (V1)
class CNNConfig:
    INPUT_CHANNELS = 2  # I and Q components
    HIDDEN_CHANNELS = 64
    NUM_RESIDUAL_BLOCKS = 4
    KERNEL_SIZE = 5


# Transformer Configuration
class UNETConfig:
    INPUT_CHANNELS = 2
    OUT_CHANNELS = 2
    FEATURES = [32, 64, 128, 256]
    # FEATURES = [64, 128, 256, 512]


# For backward compatibility with existing code
INPUT_CHANNELS = CNNConfig.INPUT_CHANNELS
HIDDEN_CHANNELS = CNNConfig.HIDDEN_CHANNELS
NUM_RESIDUAL_BLOCKS = CNNConfig.NUM_RESIDUAL_BLOCKS
KERNEL_SIZE = CNNConfig.KERNEL_SIZE

# Training
TRAIN_VAL_SPLIT = 0.8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64

# Random Seeds
RANDOM_SEED = 42
DATA_SEED = 42

# Model Selection
AVAILABLE_MODELS = ["cnn_v1", "unet"]
DEFAULT_MODEL = "unet"  # New default model
