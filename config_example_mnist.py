"""
EXAMPLE CONFIGURATION: MNIST Digit Classification

This example shows how to configure the system for MNIST digit classification.
MNIST: 70,000 images of handwritten digits (28x28 pixels), 10 classes.

Dataset: http://yann.lecun.com/exdb/mnist/
Paper used in: Defferrard et al., 2016 (NIPS spotlight experiments)

To use this configuration:
1. Download and preprocess MNIST data
2. Copy this file to config.py or import specific settings
3. Run: python run_experiment.py
"""

from config import *

# =============================================================================
# DATA SETUP: MNIST
# =============================================================================

DataConfig.DATA_DIR = './data/mnist'
DataConfig.DATA_FILE = './data/mnist/mnist_graph_features.npz'  # Preprocessed features
DataConfig.LABELS_FILE = './data/mnist/mnist_labels.npy'

# MNIST-specific preprocessing
DataConfig.TRAIN_RATIO = 0.6  # 60% train (42,000 samples)
DataConfig.VAL_RATIO = 0.2    # 20% validation (14,000 samples)
DataConfig.TEST_RATIO = 0.2   # 20% test (14,000 samples)


# =============================================================================
# GRAPH CONSTRUCTION: Grid Graph
# =============================================================================
# For MNIST, the natural graph is a 2D grid over image pixels

GraphConfig.GRAPH_TYPE = 'predefined'  # Use the 28x28 grid structure
# OR use k-NN on pixel space:
# GraphConfig.GRAPH_TYPE = 'knn'
# GraphConfig.K_NEIGHBORS = 8
# GraphConfig.KNN_METRIC = 'euclidean'

GraphConfig.NORMALIZE_LAPLACIAN = True
GraphConfig.COARSENING_LEVELS = 2  # Two levels of coarsening: 28x28 -> 14x14 -> 7x7


# =============================================================================
# MODEL ARCHITECTURE: ConvNet-style
# =============================================================================
# Simple but effective architecture for MNIST

# Convolutional layers: 32 filters -> 64 filters with pooling
ModelConfig.F_FILTERS = [32, 64]  # Number of filters per layer
ModelConfig.K_POLYNOMIAL_ORDERS = [25, 25]  # Polynomial approximation order
ModelConfig.P_POOLING_SIZES = [4, 4]  # Spatial downsampling: 4x each layer

# Fully connected layer before softmax
ModelConfig.M_FC_LAYERS = [512]  # Single hidden FC layer with 512 units
                                 # Output layer with 10 units added automatically

ModelConfig.FILTER_TYPE = 'chebyshev5'  # Fast Chebyshev polynomial filters
ModelConfig.DROPOUT_FC = 0.5  # Dropout in fully connected layer


# =============================================================================
# TRAINING: MNIST Schedule
# =============================================================================

TrainingConfig.NUM_EPOCHS = 20  # Usually converges within 20 epochs
TrainingConfig.BATCH_SIZE = 100  # Standard batch size

# Learning rate schedule: exponential decay
TrainingConfig.LEARNING_RATE_INITIAL = 0.02  # Start with small learning rate
TrainingConfig.LEARNING_RATE_DECAY_RATE = 0.95  # Decay: 0.02 * 0.95^t
TrainingConfig.LEARNING_RATE_DECAY_STEPS = 420  # Number of samples / batch_size
                                                # MNIST train: 42000 / 100 = 420

TrainingConfig.MOMENTUM = 0.9  # Standard momentum
TrainingConfig.EVAL_FREQUENCY = 30  # Evaluate every 30 steps


# =============================================================================
# REGULARIZATION: Light
# =============================================================================

RegularizationConfig.L2_REGULARIZATION = 5e-4  # Standard L2 penalty
RegularizationConfig.INPUT_DROPOUT = 0.0  # No input dropout needed


# =============================================================================
# OUTPUT
# =============================================================================

OutputConfig.OUTPUT_DIR = './outputs/mnist'
OutputConfig.CHECKPOINT_DIR = './outputs/mnist/checkpoints'
OutputConfig.SUMMARY_DIR = './outputs/mnist/summaries'
OutputConfig.VERBOSE = True


# =============================================================================
# NOTES AND EXPECTED PERFORMANCE
# =============================================================================
"""
Expected Results:
- Training accuracy: > 99%
- Validation accuracy: > 98%
- Test accuracy: > 98% (depends on architecture)

Training Time:
- GPU: ~5-10 minutes
- CPU: ~30-60 minutes

Common Issues:
1. Accuracy plateaus at 95-96% -> increase K_POLYNOMIAL_ORDERS or F_FILTERS
2. Unstable training -> reduce LEARNING_RATE_INITIAL or increase MOMENTUM
3. Overfitting (val acc << train acc) -> increase L2_REGULARIZATION or DROPOUT_FC
"""

if __name__ == '__main__':
    validate_config()
    print_config()
