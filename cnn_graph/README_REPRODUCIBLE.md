# CNN on Graphs with Fast Localized Spectral Filtering - Reproducible Implementation

This directory contains a fully reproducible and configurable implementation of Convolutional Neural Networks (CNNs) applied to arbitrary graphs with fast localized spectral filtering, as presented in the paper:

**"Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"**  
Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst  
Neural Information Processing Systems (NIPS), 2016  
[[Paper](https://arxiv.org/abs/1606.09375)] [[Original Code](https://github.com/mdeff/cnn_graph)] [[Video](https://www.youtube.com/watch?v=cIA_m7vwOVQ)]

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is Graph CNN?](#what-is-graph-cnn)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Configuration Guide](#configuration-guide)
6. [Usage](#usage)
7. [Examples](#examples)
8. [Architecture Details](#architecture-details)
9. [Citation](#citation)
10. [Additional Resources](#additional-resources)

---

## Quick Start

Get started with graph CNN in 5 minutes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your experiment
# Edit one of the example config files or create your own
cp config_example_custom.py config.py
# Edit config.py with your settings

# 3. Run experiment
python run_experiment.py

# 4. View results
# Checkpoints saved to: ./outputs/checkpoints/
# TensorBoard summaries saved to: ./outputs/summaries/
```

---

## What is Graph CNN?

Graph Convolutional Neural Networks extend classical CNNs to arbitrary graph-structured data. Instead of operating on regular grids (images) or sequences (text), they work directly on:

- **Social networks** (node classification)
- **Citation networks** (document classification)
- **Molecular graphs** (property prediction)
- **Point clouds** (3D object classification)
- **Sensor networks** (anomaly detection)

### Key Innovation: Fast Localized Filters

Instead of expensive spectral convolution using full graph Fourier transform, this paper uses:

**Chebyshev polynomial approximation** of spectral filters that:
- ✅ Are K-hop localized (efficient, sparse operations)
- ✅ Avoid computing eigendecomposition of large Laplacian
- ✅ Scale to large graphs (not O(N²))
- ✅ Support fast GPU computation

---

## Key Features

### 🎛️ Fully Configurable

Everything is in a single `config.py` file with detailed comments:
- Data loading and preprocessing
- Graph construction (k-NN, predefined)
- Model architecture (layers, filters, pooling)
- Training hyperparameters (learning rate, momentum, decay)
- Regularization (L2, dropout)
- Output settings

### 📊 Example Configurations Included

Pre-configured setups for:
- [MNIST digit classification](#mnist-example)
- [20NEWS document classification](#20news-example)
- [Custom datasets](#custom-dataset-example)

### 🔬 Research-Grade Implementation

- TensorFlow computational graphs with TensorBoard logging
- Multiple spectral filtering methods (Chebyshev, Fourier, Spline)
- Graph coarsening and pooling
- Flexible architecture (1-N layers, arbitrary filter sizes)
- Batch training with validation monitoring

### 🚀 Reproducible Pipeline

- Deterministic random seeding
- Configuration snapshots
- Checkpoint and resume capabilities
- Detailed logging

---

## Installation

### Requirements

- Python 3.6+
- TensorFlow 1.x or 2.x
- NumPy, SciPy, scikit-learn
- Jupyter (optional, for notebooks)

### Setup

```bash
# Install from requirements
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Dependency Versions

```
numpy>=1.14.0
scipy>=1.0.0
scikit-learn>=0.19.0
matplotlib>=2.1.0
gensim>=3.4.0
tensorflow-gpu>=1.0.0  # or tensorflow>=1.0.0 for CPU
jupyter>=5.0.0
ipython>=5.0.0
```

---

## Configuration Guide

### Understanding the Configuration File

The configuration is organized into **6 main sections**:

#### 1. Data Configuration (`DataConfig`)

Where to load data and how to preprocess it:

```python
DataConfig.DATA_FILE = './data/mnist/features.npz'
DataConfig.LABELS_FILE = './data/mnist/labels.npy'
DataConfig.TRAIN_RATIO = 0.7
DataConfig.VAL_RATIO = 0.15
DataConfig.TEST_RATIO = 0.15
```

**For text data:**
```python
DataConfig.DOCUMENTS_FILE = './data/20news/documents.pickle'
DataConfig.CLASS_LABELS_FILE = './data/20news/labels.npy'
DataConfig.MAX_VOCAB_SIZE = 10000
DataConfig.USE_TFIDF = True
```

#### 2. Graph Configuration (`GraphConfig`)

How to construct the graph from feature space:

```python
GraphConfig.GRAPH_TYPE = 'knn'  # or 'predefined'
GraphConfig.K_NEIGHBORS = 10  # Number of nearest neighbors
GraphConfig.KNN_METRIC = 'euclidean'  # or 'cosine' for text
GraphConfig.COARSENING_LEVELS = 2  # Multi-scale pooling
GraphConfig.NORMALIZE_LAPLACIAN = True
```

#### 3. Model Configuration (`ModelConfig`)

Network architecture:

```python
# Stack of convolutional layers
ModelConfig.F_FILTERS = [32, 64]  # Filters per layer
ModelConfig.K_POLYNOMIAL_ORDERS = [20, 20]  # Polynomial degree
ModelConfig.P_POOLING_SIZES = [4, 2]  # Spatial reduction

# Fully connected layers
ModelConfig.M_FC_LAYERS = [512]  # Hidden units

# Filter type
ModelConfig.FILTER_TYPE = 'chebyshev5'  # Fast polynomial filters
```

**Architecture Example:**

```
Input (N × 1000 features)
  ↓ ChebConv+Bias+ReLU (32 filters, K=20)
  ↓ MaxPool (reduce by 4x)
  ↓ ChebConv+Bias+ReLU (64 filters, K=20)
  ↓ MaxPool (reduce by 2x)
  ↓ Flatten
  ↓ Dense (512 units, ReLU, Dropout 0.5)
  ↓ Output (num_classes units, Softmax)
```

#### 4. Training Configuration (`TrainingConfig`)

Training procedure hyperparameters:

```python
TrainingConfig.NUM_EPOCHS = 20
TrainingConfig.BATCH_SIZE = 100
TrainingConfig.LEARNING_RATE_INITIAL = 0.1
TrainingConfig.LEARNING_RATE_DECAY_RATE = 0.95
TrainingConfig.MOMENTUM = 0.9
TrainingConfig.EVAL_FREQUENCY = 30
```

#### 5. Regularization Configuration (`RegularizationConfig`)

Overfitting prevention:

```python
RegularizationConfig.L2_REGULARIZATION = 5e-4  # Weight decay
RegularizationConfig.INPUT_DROPOUT = 0.0  # Dropout on input
# ModelConfig.DROPOUT_FC = 0.5  # Dropout in FC layers (in ModelConfig)
```

#### 6. Output Configuration (`OutputConfig`)

Where to save results:

```python
OutputConfig.OUTPUT_DIR = './outputs'
OutputConfig.CHECKPOINT_DIR = './outputs/checkpoints'
OutputConfig.SUMMARY_DIR = './outputs/summaries'
OutputConfig.VERBOSE = True
OutputConfig.SAVE_BEST_MODEL = True
```

### Configuration Validation

The system automatically validates configuration:

```python
from config import validate_config
validate_config()  # Raises AssertionError if invalid
```

---

## Usage

### Basic Workflow

1. **Prepare your data**
   ```python
   # Your feature matrix (N_samples × N_features)
   # Shape: (1000, 500) for 1000 samples, 500 features
   # Save: np.save('labels.npy', y) or scipy.sparse.save_npz('features.npz', X)
   ```

2. **Choose/Create configuration**
   ```bash
   # Use example or create custom
   cp config_example_custom.py config.py
   # Edit config.py
   ```

3. **Run experiment**
   ```bash
   python run_experiment.py
   ```

4. **Monitor training**
   ```bash
   tensorboard --logdir=./outputs/summaries
   # View at http://localhost:6006
   ```

5. **Evaluate results**
   ```python
   # Test accuracy automatically printed
   # Predictions saved to ./outputs/results/
   ```

### Advanced: Direct API Usage

For more control, use the library directly:

```python
from config import *
from lib import models, graph, utils
import numpy as np

# Load and preprocess data
X = np.load('features.npz')['data']
y = np.load('labels.npy')

# Build graph
from lib.graph import grid, distance_scipy_spatial, adjacency, laplacian
W = adjacency(dist, idx)  # Weight matrix
L = laplacian(W, normalized=True)  # Laplacian

# Create model
params = {
    'num_epochs': 20,
    'learning_rate': 0.1,
    'batch_size': 100,
    'F': [32, 64],
    'K': [20, 20],
    'p': [4, 2],
    'M': [512]
}
model = models.cgcnn(L, **params)

# Train
model.fit(X_train, y_train, X_val, y_val)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
```

---

## Examples

### MNIST Example

Classify handwritten digits using 2D grid graph:

```bash
cp config_example_mnist.py config.py
python run_experiment.py
```

**Expected performance:** 98%+ test accuracy  
**Training time:** ~5 minutes (GPU), ~30 minutes (CPU)

See [config_example_mnist.py](config_example_mnist.py) for full details.

### 20NEWS Example

Classify documents into 20 newsgroups using word co-occurrence graph:

```bash
cp config_example_20news.py config.py
python run_experiment.py
```

**Expected performance:** 74-78% test accuracy  
**Training time:** ~10 minutes (GPU), ~1 hour (CPU)

See [config_example_20news.py](config_example_20news.py) for full details.

### Custom Dataset Example

Template for your own datasets:

```bash
cp config_example_custom.py config.py
# Edit with your paths and parameters
python run_experiment.py
```

See [config_example_custom.py](config_example_custom.py) for detailed instructions.

---

## Architecture Details

### Graph Construction

The first step is constructing a graph from your feature space:

#### K-Nearest Neighbors Graph

Connect each node to its K nearest neighbors:

```python
GraphConfig.GRAPH_TYPE = 'knn'
GraphConfig.K_NEIGHBORS = 10
GraphConfig.KNN_METRIC = 'euclidean'  # or 'cosine'

# Steps:
# 1. Compute pairwise distances between features
# 2. Find k nearest neighbors for each point
# 3. Connect to neighbors with Gaussian kernel weights
# 4. Symmetrize to undirected graph
# 5. Compute graph Laplacian
```

**When to use k-NN:**
- ✅ No predefined graph available
- ✅ Feature vectors are meaningful
- ✅ Discovering neighborhood from data

**Parameters:**
- `K_NEIGHBORS`: 5-20 typical. Higher K = denser graph, more computation
- `KNN_METRIC`: 'euclidean' for images/continuous, 'cosine' for text/high-dim

#### Predefined Graph

Use an existing adjacency or Laplacian matrix:

```python
GraphConfig.GRAPH_TYPE = 'predefined'
GraphConfig.ADJACENCY_FILE = 'my_adjacency.npz'
```

**When to use:**
- ✅ Graph structure is known (social networks, molecular structure)
- ✅ More efficient (no distance computation needed)

### Spectral Filtering: Chebyshev Approximation

Core operation: apply filter in Fourier domain without computing eigenvalues

```
Standard spectral convolution:
  y = U @ diag(h(Λ)) @ U^T @ x     [O(N²) cost, requires eigendecomposition]

Chebyshev approximation:
  y ≈ Σ c_k * T_k(L̂) @ x          [O(K*nnz(L)) cost, only matrix multiplications]
  
  where T_k = Chebyshev polynomial of degree k
        L̂ = rescaled Laplacian in [-1, 1]
        K = polynomial order
```

**Advantages:**
- No eigenvalue computation
- Localized (K-hop neighborhood)
- Fast GPU implementation (sparse-dense matrix multiplication)

**Parameters:**
- `K_POLYNOMIAL_ORDERS`: Higher K = more accurate filter but slower
  - Typical: 5-25
  - K=5: very fast, good for large graphs
  - K=20: good balance of accuracy and speed
  - K=50+: very accurate but slow

### Pooling and Coarsening

Multi-scale representation through graph coarsening:

```python
GraphConfig.COARSENING_LEVELS = 2

# For 28×28 grid (784 nodes):
# Level 0: 784 nodes
#   ↓ (pool by 4)
# Level 1: 196 nodes
#   ↓ (pool by 4)
# Level 2: 49 nodes
```

**Pooling operation:**
1. Assign each node to coarser level using graph partitioning (METIS)
2. Downsample signal by taking max over partition

**When to use:**
- ✅ Hierarchical structure desired
- ✅ Graph is large (coarsening reduces computation)
- ✅ Task benefits from multiple scales

---

## Hyperparameter Tuning Guide

### Starting Point

Begin with these default settings and adjust based on performance:

```python
# Model
F_FILTERS = [32]
K_POLYNOMIAL_ORDERS = [10]
P_POOLING_SIZES = [1]
DROPOUT_FC = 0.5

# Training
LEARNING_RATE_INITIAL = 0.1
NUM_EPOCHS = 50
L2_REGULARIZATION = 1e-4

# Graph
K_NEIGHBORS = 10
```

### If Model Underfits (accuracy too low)

Underfitting = training accuracy also low, not overfitting

**Actions:**
1. Increase model capacity:
   ```python
   F_FILTERS = [64, 128]  # More filters
   K_POLYNOMIAL_ORDERS = [20, 20]  # Higher degree polynomials
   ```

2. Increase learning rate slightly (0.05 → 0.2)

3. Ensure data is normalized properly

4. Check graph construction: might need different K

### If Model Overfits (train acc >> val acc)

Overfitting = training accuracy high, validation accuracy low

**Actions:**
1. Add regularization:
   ```python
   L2_REGULARIZATION = 1e-3  # Increase weight decay
   DROPOUT_FC = 0.7  # Increase dropout
   ```

2. Reduce model complexity:
   ```python
   F_FILTERS = [16]  # Fewer filters
   K_POLYNOMIAL_ORDERS = [5]  # Lower polynomials
   ```

3. Use more training data if possible

### If Training is Unstable

Unstable = loss oscillates wildly, doesn't converge smoothly

**Actions:**
1. Reduce learning rate (0.1 → 0.01)
2. Increase batch size (100 → 200)
3. Increase momentum (0.9 → 0.99)
4. Check that features are normalized (mean ~0, std ~1)

### If Training is Slow

**Actions:**
1. Increase batch size (more GPU parallelism)
2. Reduce polynomial order (5 instead of 20)
3. Reduce K_NEIGHBORS (8 instead of 15)
4. Use GPU if available
5. Single layer network for debugging

---

## Citation

If you use this code in research, please cite the original paper:

```bibtex
@inproceedings{cnn_graph,
  title = {Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author = {Defferrard, Micha\"el and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2016},
  url = {https://arxiv.org/abs/1606.09375},
}
```

---

## Additional Resources

### Papers

- **Spectral Networks:** [Bruna et al., ICLR 2014](https://arxiv.org/abs/1312.6203)
- **Graph Convolutional Networks:** [Kipf & Welling, ICLR 2017](https://arxiv.org/abs/1609.02907)
- **ChebNet Variants:** [Henaff et al., 2015](https://arxiv.org/abs/1506.05163)

### Courses & Lectures

- [A Network Tour of Data Science](https://github.com/mdeff/ntds_2016) - EPFL Master Course
- [Deep Learning on Graphs](https://arxiv.org/abs/1709.05584) - Tutorial

### Related Code

- Original implementation: https://github.com/mdeff/cnn_graph
- Graph Neural Networks: https://github.com/rusty1s/pytorch_geometric
- Spectral Methods: https://github.com/graph-based-learning/spectral_datasets

---

## Troubleshooting

### Q: How do I load my data?

**A:** See [DataConfig](#data-configuration) section. Data can be:
- NumPy arrays: `np.save()` and `np.load()`
- SciPy sparse: `scipy.sparse.save_npz()` and `scipy.sparse.load_npz()`
- Pickle files: `pickle.dump()` and `pickle.load()`

### Q: How do I interpret K_POLYNOMIAL_ORDERS?

**A:** It's the degree of the Chebyshev polynomial. Think of it as "filter complexity":
- K=5: Very simple, fast
- K=10: Medium
- K=20: More complex, slower
- K=50: Very complex, quite slow

Higher K means the filter can have more oscillations and better approximate arbitrary spectral responses. For most problems, K=20 is good.

### Q: What should my graph be?

**A:** The graph should encode similarity between your samples:
- For images: use k-NN in pixel space or predefined grid
- For text: use k-NN in word vector space (cosine distance)
- For social networks: use the network structure itself
- For molecules: use the molecular structure graph

General principle: nodes that should be "close" in the graph should have similar features or belong to same class.

### Q: Why is accuracy not improving?

**A:** Check these in order:
1. Data loaded correctly (shapes, no NaNs)
2. Labels are integers 0 to num_classes-1
3. Graph construction working (connected components > 1)
4. Learning rate appropriate for your scale
5. Model capacity sufficient (try larger F_FILTERS)

### Q: How do I use GPU?

**A:** Just install tensorflow-gpu instead of tensorflow:
```bash
pip install tensorflow-gpu
```

TensorFlow will automatically detect and use GPU. No code changes needed!

### Q: Can I use this for my dataset?

**A:** Very likely yes! The method works for any dataset where:
- You have feature vectors (or can construct them)
- Neighbors in feature space are meaningful
- Classification/regression problem

Common applications: document classification, image classification, graph classification, link prediction.

---

## License

This implementation is released under the MIT License. See [LICENSE.txt](LICENSE.txt).

The original paper and code are from Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst.

---

## Contact & Questions

For issues or questions about:
- **This implementation:** Create an issue on GitHub
- **Original paper:** See https://github.com/mdeff/cnn_graph
- **Graph neural networks in general:** See PyTorch Geometric documentation

Happy graph learning! 🎉
