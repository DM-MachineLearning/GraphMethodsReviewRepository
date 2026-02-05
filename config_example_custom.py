"""
EXAMPLE CONFIGURATION: Custom Dataset Template

Use this template to configure the system for your own dataset.
This is a flexible starting point you can customize.

To use this configuration:
1. Prepare your data:
   - Feature matrix: shape (N_samples, N_features)
   - Labels: shape (N_samples,) with values 0 to N_classes-1
   - Save as numpy/scipy sparse matrices
   
2. Customize the parameters below for your problem
3. Run: python run_experiment.py

"""

from config import *

# =============================================================================
# STEP 1: LOAD YOUR DATA
# =============================================================================

# Option A: Preprocessed numpy/sparse matrices
DataConfig.DATA_DIR = './data/my_dataset'
DataConfig.DATA_FILE = './data/my_dataset/features.npz'  # scipy.sparse.save_npz()
DataConfig.LABELS_FILE = './data/my_dataset/labels.npy'  # np.save()

# Option B: Raw text documents (uncomment if using text)
# DataConfig.DOCUMENTS_FILE = './data/my_dataset/documents.pickle'
# DataConfig.CLASS_LABELS_FILE = './data/my_dataset/labels.npy'
# DataConfig.NUM_HANDLING = 'substitute'
# DataConfig.MAX_VOCAB_SIZE = 5000
# DataConfig.USE_TFIDF = True

# Train/Val/Test split
DataConfig.TRAIN_RATIO = 0.7
DataConfig.VAL_RATIO = 0.15
DataConfig.TEST_RATIO = 0.15


# =============================================================================
# STEP 2: GRAPH CONSTRUCTION
# =============================================================================

# Choose: 'knn' to build k-NN graph, or 'predefined' if you have adjacency matrix

# Option A: Build k-NN graph from features
GraphConfig.GRAPH_TYPE = 'knn'
GraphConfig.K_NEIGHBORS = 10  # TODO: Tune this (8-15 typical)
GraphConfig.KNN_METRIC = 'euclidean'  # 'euclidean' for images/signals, 'cosine' for text
GraphConfig.NORMALIZE_FEATURES = True  # Usually recommended

# Option B: Use predefined adjacency matrix (uncomment if you have one)
# GraphConfig.GRAPH_TYPE = 'predefined'
# GraphConfig.ADJACENCY_FILE = './data/my_dataset/adjacency.npz'

# Graph properties
GraphConfig.NORMALIZE_LAPLACIAN = True
GraphConfig.COARSENING_LEVELS = 0  # Start with 0, increase for hierarchical pooling


# =============================================================================
# STEP 3: MODEL ARCHITECTURE - START SIMPLE, THEN TUNE
# =============================================================================

# Start with minimal architecture:
# - Single convolutional layer
# - One fully connected layer
# - Monitor training curves to decide if more capacity needed

ModelConfig.FILTER_TYPE = 'chebyshev5'  # Recommended: fast and accurate

# Minimal architecture (baseline)
ModelConfig.F_FILTERS = [32]  # Start with 32 filters
ModelConfig.K_POLYNOMIAL_ORDERS = [10]  # Lower values = faster
ModelConfig.P_POOLING_SIZES = [1]  # No pooling for now

ModelConfig.M_FC_LAYERS = [100]  # One hidden layer, adjust for your task
                                # Last element = number of classes

# TODO: If underfitting, try:
# ModelConfig.F_FILTERS = [32, 64]  # Add more layers
# ModelConfig.K_POLYNOMIAL_ORDERS = [20, 20]  # Increase polynomial order
# ModelConfig.P_POOLING_SIZES = [2, 2]  # Add pooling

ModelConfig.DROPOUT_FC = 0.5  # Enable if overfitting


# =============================================================================
# STEP 4: TRAINING PARAMETERS
# =============================================================================

# Start with conservative values
TrainingConfig.NUM_EPOCHS = 50  # Increase if not converged
TrainingConfig.BATCH_SIZE = 128  # Or 64/100 depending on GPU memory

# Learning rate: critical hyperparameter
# Start here and adjust based on training curves
TrainingConfig.LEARNING_RATE_INITIAL = 0.1  # TODO: Tune (0.01-1.0 typical)
TrainingConfig.LEARNING_RATE_DECAY_RATE = 0.95
TrainingConfig.MOMENTUM = 0.9

TrainingConfig.EVAL_FREQUENCY = 50  # Check progress frequently


# =============================================================================
# STEP 5: REGULARIZATION - ADJUST BASED ON OVERFITTING
# =============================================================================

# If validation accuracy << training accuracy: increase regularization
RegularizationConfig.L2_REGULARIZATION = 1e-4  # TODO: Tune (1e-5 to 1e-3)

# If training is unstable/noisy: enable early stopping
RegularizationConfig.EARLY_STOPPING_ENABLED = True
RegularizationConfig.EARLY_STOPPING_PATIENCE = 10


# =============================================================================
# STEP 6: OUTPUT SETTINGS
# =============================================================================

OutputConfig.OUTPUT_DIR = './outputs/my_experiment'
OutputConfig.VERBOSE = True  # Print training progress


# =============================================================================
# QUICK TUNING GUIDE
# =============================================================================
"""
COMMON PROBLEMS AND SOLUTIONS:

Problem: Accuracy is low (not converging)
Solution:
  1. Check data: ensure labels are 0 to N_classes-1
  2. Increase model capacity:
     - Add more filters: F_FILTERS = [64, 128]
     - Increase polynomial order: K_POLYNOMIAL_ORDERS = [20, 20]
  3. Increase learning rate slightly
  4. Train for more epochs

Problem: Training is unstable (loss jumps around)
Solution:
  1. Reduce LEARNING_RATE_INITIAL
  2. Increase BATCH_SIZE
  3. Check that features are normalized
  4. Reduce number of filters temporarily for debugging

Problem: Overfitting (train acc >> val acc)
Solution:
  1. Increase L2_REGULARIZATION (try 1e-3 or 1e-2)
  2. Increase DROPOUT_FC (try 0.5 or 0.7)
  3. Reduce model size (fewer filters, lower polynomial order)
  4. Use more training data if possible

Problem: Training is very slow
Solution:
  1. Increase BATCH_SIZE
  2. Reduce K_POLYNOMIAL_ORDERS (use 5 instead of 20)
  3. Reduce K_NEIGHBORS (use 5 instead of 10)
  4. Use GPU if available
  5. Reduce number of filters

DATASET-SPECIFIC PARAMETERS:

For small datasets (< 1000 samples):
  - More regularization: L2 = 1e-3 to 1e-2
  - Smaller network: F_FILTERS = [16]
  - Higher dropout: DROPOUT_FC = 0.7

For large graphs (> 10000 nodes):
  - Single layer: F_FILTERS = [32]
  - Lower polynomial: K_POLYNOMIAL_ORDERS = [5]
  - Larger batch: BATCH_SIZE = 256

For high-dimensional data (> 5000 features):
  - Use 'cosine' distance
  - Set NORMALIZE_FEATURES = True
  - Reduce K_NEIGHBORS (use 5-8)
"""

if __name__ == '__main__':
    validate_config()
    print_config()
