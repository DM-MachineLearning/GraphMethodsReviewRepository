"""
Base Configuration - CNN on Graphs with Fast Localized Spectral Filtering

Modify these parameters before running experiments.
"""

import os
import numpy as np
from typing import List, Dict, Any

# ============================================================================= 
# DATA CONFIGURATION
# =============================================================================

class DataConfig:
    DATA_FILE = './data/features.npz'
    LABELS_FILE = './data/labels.npy'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    NORMALIZE_FEATURES = True
    RANDOM_SEED = 42


# =============================================================================
# GRAPH CONFIGURATION
# =============================================================================

class GraphConfig:
    GRAPH_TYPE = 'knn'  # 'knn' or 'grid' or 'predefined'
    K_NEIGHBORS = 10
    KNN_METRIC = 'euclidean'
    NORMALIZE_LAPLACIAN = True
    COARSENING_LEVELS = 2


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelConfig:
    F_FILTERS = [32, 64]
    K_POLYNOMIAL_ORDERS = [20, 20]
    P_POOLING_SIZES = [4, 2]
    M_FC_LAYERS = [512]
    ACTIVATION = 'relu'
    BRELU = True
    POOL_TYPE = 'average'
    DROPOUT_FC = 0.5


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig:
    NUM_EPOCHS = 20
    BATCH_SIZE = 100
    LEARNING_RATE_INITIAL = 0.1
    LEARNING_RATE_DECAY_RATE = 0.95
    LEARNING_RATE_DECAY_STEPS = 100
    MOMENTUM = 0.9
    EVAL_FREQUENCY = 30


# =============================================================================
# REGULARIZATION CONFIGURATION
# =============================================================================

class RegularizationConfig:
    L2_REGULARIZATION = 5e-4
    INPUT_DROPOUT = 0.0
    EARLY_STOPPING_PATIENCE = 100


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

class OutputConfig:
    OUTPUT_DIR = './outputs'
    VERBOSE = True
    SAVE_CHECKPOINTS = True


def validate_config() -> bool:
    """Validate configuration parameters."""
    assert abs(DataConfig.TRAIN_RATIO + DataConfig.VAL_RATIO + DataConfig.TEST_RATIO - 1.0) < 1e-6
    assert len(ModelConfig.F_FILTERS) == len(ModelConfig.K_POLYNOMIAL_ORDERS)
    assert len(ModelConfig.F_FILTERS) == len(ModelConfig.P_POOLING_SIZES)
    assert GraphConfig.K_NEIGHBORS > 0
    assert TrainingConfig.NUM_EPOCHS > 0
    assert 0 <= ModelConfig.DROPOUT_FC <= 1
    return True


if __name__ == '__main__':
    validate_config()
    print("✓ Config valid")
