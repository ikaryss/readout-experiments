"""Configuration parameters for the denoising models."""

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

# Model Architectures


# Original CNN (V1)
class CNNConfig:
    INPUT_CHANNELS = 2  # I and Q components
    HIDDEN_CHANNELS = 64
    NUM_RESIDUAL_BLOCKS = 4
    KERNEL_SIZE = 5


# Transformer Configuration
class TransformerConfig:
    INPUT_CHANNELS = 2
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1


# WaveletUNet Configuration
class WaveletUNetConfig:
    INPUT_CHANNELS = 2
    HIDDEN_CHANNELS = 64
    NUM_BLOCKS = 4
    KERNEL_SIZES = [3, 5, 7]


# Hybrid Configuration
class HybridConfig:
    INPUT_CHANNELS = 2
    HIDDEN_CHANNELS = 64
    D_MODEL = 128
    NHEAD = 8
    NUM_TRANSFORMER_LAYERS = 4
    NUM_WAVELET_BLOCKS = 3


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
AVAILABLE_MODELS = ["cnn_v1", "transformer", "wavelet_unet", "hybrid"]
DEFAULT_MODEL = "hybrid"  # New default model
