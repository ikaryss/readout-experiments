"""Configuration parameters for the denoising models."""

import numpy as np

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("PyTorch not found. Please install it with: pip install torch")
    DEVICE = "cpu"

# Data Generation
MEAS_TIME = np.arange(0, 1e-6 + 12 * 2e-9, 2e-9)  # 512
SEQUENCE_LENGTH = len(MEAS_TIME)
QUBIT = 0

# IQ Parameters
CLEAN_AMP = 1
T1_TIME = 50e-6
RELAX_TRANSITION_TIME = 1e-9


# curriculum_stages
# class CurriculumStages:
#     # IN_PHASE_RANGES = [
#     #     [-100, 100],
#     #     [-250, 250],
#     #     [-500, 500],
#     #     [-750, 750],
#     #     [-1000, 1000],
#     # ]
#     # QUADRATURE_RANGES = [
#     #     [-100, 100],
#     #     [-250, 250],
#     #     [-500, 500],
#     #     [-750, 750],
#     #     [-1000, 1000],
#     # ]
#     IN_PHASE_RANGES = [[-1000, 1000]] * 10
#     QUADRATURE_RANGES = [[-1000, 1000]] * 10
#     STAGES_NOISE_AMP = [
#         [50, 300],
#         [200, 400],
#         [300, 500],
#         [400, 700],
#         [600, 800],
#         [700, 900],
#         [800, 1000],
#         [900, 1100],
#         [1000, 1300],
#         [1200, 1600],
#     ]
#     BATCHES_GROUND = [
#         50_000,
#         50_000,
#         70_000,
#         100_000,
#         100_000,
#         200_000,
#         200_000,
#         200_000,
#         200_000,
#         200_000,
#     ]
#     BATCHES_EXCITED = [
#         50_000,
#         50_000,
#         70_000,
#         100_000,
#         100_000,
#         200_000,
#         200_000,
#         200_000,
#         200_000,
#         200_000,
#     ]
#     BATCHES_RELAX = [
#         80_000,
#         80_000,
#         90_000,
#         140_000,
#         140_000,
#         240_000,
#         260_000,
#         260_000,
#         260_000,
#         260_000,
#     ]
#     # BATCHES_GROUND = [3_000] * 5s
#     # BATCHES_EXCITED = [3_000] * 5
#     # BATCHES_RELAX = [3_000] * 5
#     # EPOCHS = [15, 20, 40, 60, 100]
#     EPOCHS = [30] * 10


class CurriculumStages:
    IN_PHASE_RANGES = [[-5000, 5000]]
    QUADRATURE_RANGES = [[-5000, 5000]]
    STAGES_NOISE_AMP = [[1000, 1500]]
    BATCHES_GROUND = [15_000]
    BATCHES_EXCITED = [15_000]
    BATCHES_RELAX = [15_000]
    EPOCHS = [15]


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
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256

# Random Seeds
RANDOM_SEED = 42
DATA_SEED = 42

# Model Selection
AVAILABLE_MODELS = ["cnn_v1", "unet"]
DEFAULT_MODEL = "unet"  # New default model
