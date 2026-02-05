"""
EXAMPLE CONFIGURATION: 20NEWS Text Classification

This example shows how to configure the system for 20NEWS document classification.
20NEWS: ~18,000 documents from 20 newsgroups

Dataset: http://qwone.com/~jason/20Newsgroups/
Paper used in: Defferrard et al., 2016 (NIPS experiments)

The pipeline:
1. Raw documents -> word frequency vectors
2. Construct word co-occurrence graph from k-NN in vector space
3. Apply graph CNN to classify documents into newsgroups

To use this configuration:
1. Download 20NEWS dataset
2. Preprocess to word frequency vectors
3. Run: python run_experiment.py
"""

from config import *

# =============================================================================
# DATA SETUP: 20NEWS
# =============================================================================

DataConfig.DATA_DIR = './data/20news'

# Raw text documents and labels
DataConfig.DOCUMENTS_FILE = './data/20news/20news.pickle'  # Documents
DataConfig.CLASS_LABELS_FILE = './data/20news/20news_labels.npy'  # Class indices

# Or preprocessed data
DataConfig.DATA_FILE = './data/20news/20news_tfidf.npz'  # Sparse TF-IDF matrix
DataConfig.LABELS_FILE = './data/20news/20news_labels.npy'

# Text preprocessing parameters
DataConfig.NUM_HANDLING = 'substitute'  # Convert numbers to NUM token
DataConfig.MAX_VOCAB_SIZE = 10000  # Keep top 10k words
DataConfig.MIN_WORD_FREQ = 5  # Ignore words appearing < 5 times
DataConfig.USE_TFIDF = True  # Use TF-IDF instead of raw counts

# Train/Val/Test split
DataConfig.TRAIN_RATIO = 0.7  # 70% training (~12,600 docs)
DataConfig.VAL_RATIO = 0.15   # 15% validation (~2,700 docs)
DataConfig.TEST_RATIO = 0.15  # 15% test (~2,700 docs)


# =============================================================================
# GRAPH CONSTRUCTION: Word Co-occurrence Graph
# =============================================================================
# Build graph from word similarity in vector space

GraphConfig.GRAPH_TYPE = 'knn'  # Construct k-NN graph from word vectors

# K-NN in word vector space
GraphConfig.K_NEIGHBORS = 10  # Connect each word to 10 most similar words
GraphConfig.KNN_METRIC = 'cosine'  # Cosine distance for text (standard)
GraphConfig.NORMALIZE_FEATURES = True  # Normalize TF-IDF vectors

# Gaussian kernel parameters
GraphConfig.SIGMA_SCALING = 1.0  # Standard bandwidth

# Laplacian properties
GraphConfig.NORMALIZE_LAPLACIAN = True  # Normalized Laplacian (standard for graphs)
GraphConfig.COARSENING_LEVELS = 1  # One level of coarsening


# =============================================================================
# MODEL ARCHITECTURE: Single-layer CNN
# =============================================================================
# Simpler architecture for text classification

ModelConfig.FILTER_TYPE = 'chebyshev5'  # Chebyshev polynomial filters
ModelConfig.F_FILTERS = [32]  # Single convolutional layer with 32 filters
ModelConfig.K_POLYNOMIAL_ORDERS = [5]  # Lower polynomial order for efficiency
ModelConfig.P_POOLING_SIZES = [1]  # No spatial pooling in text case

# Fully connected layers
ModelConfig.M_FC_LAYERS = [100]  # Single hidden layer with 100 units
                                # Output layer: 20 units (one per newsgroup)

# Activation and regularization
ModelConfig.ACTIVATION_LAYER1 = 'b1relu'
ModelConfig.DROPOUT_FC = 1.0  # NO dropout (1.0 = keep all units)
                              # Uncomment below to enable dropout
# ModelConfig.DROPOUT_FC = 0.5


# =============================================================================
# TRAINING: 20NEWS Schedule
# =============================================================================

TrainingConfig.NUM_EPOCHS = 50  # More epochs for convergence on text
TrainingConfig.BATCH_SIZE = 128  # Larger batch for text data

# Learning rate schedule
TrainingConfig.LEARNING_RATE_INITIAL = 0.1  # Higher learning rate for text
TrainingConfig.LEARNING_RATE_DECAY_RATE = 0.999  # Slower decay
TrainingConfig.LEARNING_RATE_DECAY_STEPS = None  # Auto: one epoch worth

TrainingConfig.MOMENTUM = 0.9
TrainingConfig.EVAL_FREQUENCY = 5  # Evaluate frequently


# =============================================================================
# REGULARIZATION: Moderate
# =============================================================================
# Text data can be high-dimensional and noisy

RegularizationConfig.L2_REGULARIZATION = 1e-3  # Moderate L2 regularization
RegularizationConfig.INPUT_DROPOUT = 0.0  # No input dropout


# =============================================================================
# OUTPUT
# =============================================================================

OutputConfig.OUTPUT_DIR = './outputs/20news'
OutputConfig.CHECKPOINT_DIR = './outputs/20news/checkpoints'
OutputConfig.SUMMARY_DIR = './outputs/20news/summaries'
OutputConfig.VERBOSE = True


# =============================================================================
# NOTES AND EXPECTED PERFORMANCE
# =============================================================================
"""
Expected Results:
- Training accuracy: 85-90%
- Validation accuracy: 75-80%
- Test accuracy: 74-78% (depends on architecture)

Training Time:
- GPU: ~5-15 minutes
- CPU: ~1-2 hours

Common Issues:
1. Accuracy low (< 70%) -> increase K_NEIGHBORS or F_FILTERS
2. Overfitting (large gap between train/val) -> increase L2_REGULARIZATION
3. Too slow -> reduce K_NEIGHBORS, F_FILTERS, or K_POLYNOMIAL_ORDERS
4. Unstable training -> reduce LEARNING_RATE_INITIAL

Hyperparameter Tuning Tips:
- Start with K_NEIGHBORS=8-15 (too low = disconnected graph, too high = dense)
- K_POLYNOMIAL_ORDERS=5 is usually sufficient for text (more = slower)
- F_FILTERS=[32-64] works well (increase if underfitting)
- Learning rate 0.1 is typical for momentum optimizer with text
"""

if __name__ == '__main__':
    validate_config()
    print_config()
