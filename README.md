# GraphMethodsReviewRepository

A comprehensive, reproducible repository of research papers implementing graph neural network methods and spectral filtering techniques for machine learning.

This repository is designed to be **fully reproducible, scalable, and easy to use**. Each method is in its own folder with complete configuration files, examples, and documentation.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Methods Overview](#methods-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)

---

## Overview

This repository contains implementations of seminal papers on graph neural networks and spectral filtering methods. Each method includes:

- ✅ **Complete Implementation** - Fully functional, tested code
- ✅ **Configuration System** - All parameters in easy-to-edit `config.py` files
- ✅ **Example Configurations** - Pre-configured for MNIST, 20NEWS, custom data
- ✅ **Detailed Documentation** - Architecture explanations and usage guides
- ✅ **Reproducible Pipelines** - Deterministic, logged, versioned experiments

**Philosophy:** Make cutting-edge research immediately usable for your specific problem.

---

## Directory Structure

```
GraphMethodsReviewRepository/
├── README.md (this file)
├── cnn_graph/                 ← CNN on Graphs with Spectral Filtering
│   ├── README_REPRODUCIBLE.md ← Full documentation - START HERE!
│   ├── config.py              ← Main configuration file
│   ├── config_example_*.py    ← Example configurations
│   ├── run_experiment.py      ← Main execution script
│   ├── lib/                   ← Core implementation
│   └── [notebooks/examples/]
└── [additional methods...]
```

---

## Methods Overview

### 1. **CNN on Graphs with Fast Localized Spectral Filtering**

📄 **Paper:** Defferrard et al., NIPS 2016  
🔗 **Folder:** [`cnn_graph/`](cnn_graph/)  
📚 **Documentation:** [`cnn_graph/README_REPRODUCIBLE.md`](cnn_graph/README_REPRODUCIBLE.md)

**Key Innovation:** Fast spectral convolution on arbitrary graphs using Chebyshev polynomial approximation.

**Status:** ✅ Fully Implemented

**Features:**
- Graph convolutional layers with spectral filters
- K-NN graph construction
- Graph coarsening and multi-scale pooling
- TensorBoard logging
- Pre-configured examples for MNIST, 20NEWS, custom data

**Performance:**
- MNIST: 98-99% accuracy
- 20NEWS: 74-78% accuracy

---

## Quick Start

### 3-Minute Setup

```bash
# 1. Enter method directory
cd cnn_graph

# 2. Copy example configuration
cp config_example_custom.py config.py

# 3. Edit with your data paths and parameters
nano config.py

# 4. Install and run
pip install -r requirements.txt
python run_experiment.py
```

**That's it!** Results saved to `./outputs/`

---

## Usage Examples

### MNIST Digit Classification

```bash
cd cnn_graph
cp config_example_mnist.py config.py
python run_experiment.py
```

See [`config_example_mnist.py`](cnn_graph/config_example_mnist.py) for full details.

### 20NEWS Text Classification

```bash
cd cnn_graph
cp config_example_20news.py config.py
python run_experiment.py
```

See [`config_example_20news.py`](cnn_graph/config_example_20news.py) for full details.

### Custom Dataset

```bash
cd cnn_graph
cp config_example_custom.py config.py
# Edit config.py with your data paths
python run_experiment.py
```

See [`config_example_custom.py`](cnn_graph/config_example_custom.py) for guided setup.

---

## Installation

### Requirements

- Python 3.6+
- TensorFlow 1.x or 2.x
- NumPy, SciPy, scikit-learn

### Setup

```bash
# Clone and enter directory
git clone <repo>
cd GraphMethodsReviewRepository/cnn_graph

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration Overview

All parameters are in `config.py`:

```python
# Data
DataConfig.DATA_FILE = './data/features.npz'
DataConfig.LABELS_FILE = './data/labels.npy'
DataConfig.TRAIN_RATIO = 0.7

# Graph construction
GraphConfig.GRAPH_TYPE = 'knn'
GraphConfig.K_NEIGHBORS = 10
GraphConfig.NORMALIZE_FEATURES = True

# Model architecture
ModelConfig.F_FILTERS = [32, 64]
ModelConfig.K_POLYNOMIAL_ORDERS = [20, 20]
ModelConfig.P_POOLING_SIZES = [4, 2]

# Training
TrainingConfig.NUM_EPOCHS = 20
TrainingConfig.LEARNING_RATE_INITIAL = 0.1
TrainingConfig.BATCH_SIZE = 100

# Regularization
RegularizationConfig.L2_REGULARIZATION = 5e-4
ModelConfig.DROPOUT_FC = 0.5
```

Full documentation with detailed parameter descriptions: [`cnn_graph/README_REPRODUCIBLE.md`](cnn_graph/README_REPRODUCIBLE.md)

---

## Output

After running, you'll find:

```
outputs/
├── checkpoints/     # Model checkpoints
├── summaries/       # TensorBoard event files
├── logs/            # Training logs
└── results/         # Results, predictions, plots
```

View training progress:
```bash
tensorboard --logdir=./outputs/summaries
```

---

## Citation

```bibtex
@inproceedings{defferrard2016cnn,
  title = {Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author = {Defferrard, Micha\"el and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2016},
}
```

---

## Resources

- 📖 [Full Documentation](cnn_graph/README_REPRODUCIBLE.md)
- 🔗 [Original Paper](https://arxiv.org/abs/1606.09375)
- 💻 [Original Code](https://github.com/mdeff/cnn_graph)

---

## License

MIT License - See individual method folders for details.

---

**Next Steps:**

1. Choose a method: `cd cnn_graph/`
2. Read full docs: `cat README_REPRODUCIBLE.md`
3. Run example: `python run_experiment.py`

Happy graph learning! ��
