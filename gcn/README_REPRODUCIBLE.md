# GCN - Graph Convolutional Networks Implementation Guide

## Overview

This is a reproducible implementation of **Graph Convolutional Networks (GCN)** for semi-supervised node classification on graph-structured data.

**Paper:** Kipf & Welling (2017) - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)

**Key Contributions:**
- Scalable approach for semi-supervised learning on graphs
- Efficient localized first-order spectral graph convolutions
- Linear complexity in number of graph edges
- State-of-the-art results on citation networks

---

## Quick Start

### 1. Installation

```bash
# Navigate to GCN directory
cd gcn

# Install dependencies
pip install -r requirements.txt

# Or use setup
python setup.py install
```

### 2. Run on Cora Dataset (Paper Benchmark)

```bash
# Using the reproducible pipeline
cd ..  # Back to root
python run_experiment.py

# Or use the original Kipf implementation
cd gcn
python train.py --dataset cora
```

### 3. Expected Output

```
✓ Data loaded: 2,708 nodes, 5,429 edges, 1,433 features
✓ Model built: GCN with 16 hidden units
✓ Training for 200 epochs...
✓ Best validation accuracy: ~81.5%
✓ Test accuracy: ~81.5%
✓ Results saved to outputs/results/
```

---

## Dataset Information

### Available Datasets

Three citation networks from [Planetoid](https://github.com/kimiyoung/planetoid):

| Dataset | Nodes | Edges | Features | Classes | Test Nodes |
|---------|-------|-------|----------|---------|-----------|
| **Cora** | 2,708 | 5,429 | 1,433 | 7 | 1,000 |
| **Citeseer** | 3,327 | 4,732 | 3,703 | 6 | 1,000 |
| **Pubmed** | 19,717 | 44,338 | 500 | 3 | 1,000 |

### Data Format

Each dataset (e.g., `cora`) includes files:
```
ind.cora.x       - 2,708 × 1,433 sparse feature matrix
ind.cora.y       - 2,708 × 7 one-hot labels (all data)
ind.cora.tx      - 1,000 × 1,433 test feature matrix
ind.cora.ty      - 1,000 × 7 test labels
ind.cora.allx    - 2,708 × 1,433 all labeled features
ind.cora.ally    - 2,708 × 7 all labels
ind.cora.graph   - Graph structure (pickled dict)
ind.cora.test.index - Test indices
```

**Data splits:**
- Training: 140 nodes (fixed per paper)
- Validation: 500 nodes
- Test: ~1,000 nodes

---

## Configuration System

### 1. Configuration Files

All parameters defined in `config.py` with validation:

**DataConfig** - Dataset selection and preprocessing
```python
dataset_name: str = 'cora'           # 'cora', 'citeseer', 'pubmed'
dataset_path: str = './gcn/data'     # Path to data files
normalize_features: bool = True      # Feature normalization
sparse_features: bool = True         # Use sparse matrices
```

**GraphConfig** - Graph preprocessing
```python
add_self_loops: bool = True          # Add self-connections
normalize_adj: bool = True           # Normalize adjacency
symmetric_normalize: bool = True     # Symmetric normalization
max_degree: int = 3                  # For Chebyshev polynomials
```

**ModelConfig** - Architecture parameters (from paper)
```python
model_type: str = 'gcn'              # 'gcn', 'gcn_cheby', 'dense'
hidden1: int = 16                    # Hidden layer units
dropout: float = 0.5                 # Dropout rate
weight_decay: float = 5e-4           # L2 regularization
activation: str = 'relu'             # Activation function
```

**TrainingConfig** - Learning parameters (from paper)
```python
epochs: int = 200                    # Training epochs
learning_rate: float = 0.01          # Learning rate
optimizer: str = 'adam'              # Optimizer type
early_stopping: int = 10             # Patience for early stopping
seed: int = 123                      # Random seed
```

**OutputConfig** - Output management
```python
output_dir: str = './outputs'        # Output directory
save_model: bool = True              # Save trained model
save_results: bool = True            # Save results JSON
verbose: bool = True                 # Verbose logging
```

### 2. Example Configurations

Pre-configured templates for different datasets:

```bash
# Cora dataset (default)
python -c "from config_example_cora import config; config.validate()"

# Citeseer dataset
python -c "from config_example_citeseer import config; config.validate()"
```

### 3. Using Custom Configuration

```python
from config import GCNConfig, DataConfig, ModelConfig, TrainingConfig

# Create custom config
config = GCNConfig(
    data=DataConfig(dataset_name='citeseer'),
    model=ModelConfig(hidden1=32, dropout=0.6),
    training=TrainingConfig(epochs=300, learning_rate=0.005)
)

# Validate
config.validate()

# Use in training
from run_experiment import GCNExperiment
experiment = GCNExperiment(config)
experiment.run()
```

---

## Data Parameters & Paths

### Data Paths

```
gcn/
  └─ data/
      ├─ ind.cora.{x, y, tx, ty, allx, ally, graph, test.index}
      ├─ ind.citeseer.{...}
      └─ ind.pubmed.{...}
```

### Parameter Values (Paper Settings)

| Component | Parameter | Cora | Citeseer | Pubmed |
|-----------|-----------|------|----------|--------|
| **Graph** | Nodes | 2,708 | 3,327 | 19,717 |
| | Edges | 5,429 | 4,732 | 44,338 |
| | Features | 1,433 | 3,703 | 500 |
| | Classes | 7 | 6 | 3 |
| **Model** | Hidden1 | 16 | 16 | 16 |
| | Dropout | 0.5 | 0.5 | 0.5 |
| | Weight Decay | 5e-4 | 5e-4 | 5e-4 |
| **Training** | Epochs | 200 | 200 | 200 |
| | Learning Rate | 0.01 | 0.01 | 0.01 |
| | Early Stopping | 10 | 10 | 10 |
| | Optimizer | Adam | Adam | Adam |

### Expected Results (from Paper)

| Dataset | Test Accuracy | Standard Deviation |
|---------|---------------|-------------------|
| **Cora** | 81.5% | ±0.5% |
| **Citeseer** | 70.3% | ±0.7% |
| **Pubmed** | 79.0% | ±0.3% |

---

## Model Architecture

### GCN Layer

The core GCN layer is defined as:

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

Where:
- $\tilde{A} = A + I$ (adjacency with self-loops)
- $\tilde{D}$ is the degree matrix of $\tilde{A}$
- $H^{(l)}$ is the hidden layer representation
- $W^{(l)}$ is the weight matrix
- $\sigma$ is ReLU activation

### Two-Layer GCN for Semi-Supervised Classification

```
Input Features (N × D)
         ↓
  GCN Layer 1
  16 hidden units
  ReLU activation
  Dropout (0.5)
         ↓
  GCN Layer 2
  7/6/3 output units (per dataset)
  Softmax activation
         ↓
Output Predictions (N × C)
```

### Training Details

- **Loss Function:** Cross-entropy with L2 regularization
- **Masking:** Only training nodes contribute to loss
- **Validation:** Monitor on validation set (early stopping)
- **Regularization:** Weight decay λ = 5×10⁻⁴
- **Dropout:** 50% during training, 0% during evaluation

---

## Running Experiments

### Method 1: Reproducible Pipeline (Recommended)

```bash
# Run complete end-to-end pipeline
python run_experiment.py

# With custom configuration
python -c "
from config_example_cora import config
from run_experiment import GCNExperiment
exp = GCNExperiment(config)
exp.run()
"
```

### Method 2: Original Kipf Implementation

```bash
cd gcn
python train.py --dataset cora
python train.py --dataset citeseer
python train.py --dataset pubmed
```

### Method 3: Custom Script

```python
from config import get_cora_config
from run_experiment import GCNExperiment

# Get configuration
config = get_cora_config()
config.validate()

# Run experiment
exp = GCNExperiment(config)
exp.run()

# Results saved to outputs/results/
```

---

## Output & Results

### Directory Structure

```
outputs/
├── logs/
│   └── experiment_YYYYMMDD_HHMMSS.log
├── checkpoints/
│   └── model_YYYYMMDD_HHMMSS.ckpt
└── results/
    ├── results_YYYYMMDD_HHMMSS.json
    └── config_YYYYMMDD_HHMMSS.json
```

### Results JSON Format

```json
{
  "timestamp": "2024-02-06T10:30:45",
  "dataset": "cora",
  "model": "gcn",
  "results": {
    "test_accuracy": 0.815,
    "test_loss": 0.4231,
    "best_val_acc": 0.815,
    "best_val_loss": 0.4205,
    "best_epoch": 125,
    "train_losses": [...],
    "val_losses": [...],
    "val_accs": [...]
  },
  "configuration": {
    "data": {...},
    "model": {...},
    "training": {...}
  }
}
```

---

## Reproducibility

### Ensuring Reproducible Results

1. **Random Seed:**
   ```python
   config.training.seed = 123  # Set in TrainingConfig
   ```

2. **TensorFlow Settings:**
   ```python
   tf.set_random_seed(seed)
   np.random.seed(seed)
   ```

3. **Full Batch Training:**
   ```python
   config.training.batch_size = None  # Use full batch
   ```

4. **Same Dataset Split:**
   - Uses Planetoid dataset splits (fixed in paper)
   - Exactly 140 training nodes per dataset

### Verification

Run multiple experiments with same config:

```bash
for i in {1..5}; do
  python run_experiment.py
done

# Compare outputs in outputs/results/
ls -la outputs/results/
```

Expected: Within ±1% accuracy variation between runs.

---

## Troubleshooting

### Issue: "No module named 'gcn'"

```bash
# Solution: Ensure proper path setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)/gcn"
python run_experiment.py
```

### Issue: "Dataset not found"

```bash
# Solution: Check data files exist
ls -la gcn/data/ind.cora.*

# If missing, re-download from Kipf's repo
# https://github.com/tkipf/gcn/tree/master/gcn/data
```

### Issue: Low accuracy (< 70%)

- Check random seed is set to 123
- Verify all parameters match paper settings
- Ensure full batch training (batch_size=None)
- Check for TensorFlow version compatibility

### Issue: TensorFlow version mismatch

```bash
# Check version
python -c "import tensorflow; print(tensorflow.__version__)"

# Requirement: TensorFlow 1.15.4 (legacy) or compatible
# For TensorFlow 2.x, may need code updates
```

---

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration system with validation |
| `run_experiment.py` | End-to-end training pipeline |
| `data_loader.py` | Data loading utilities |
| `config_example_*.py` | Pre-configured examples |
| `gcn/train.py` | Original Kipf implementation |
| `gcn/models.py` | GCN/MLP model classes |
| `gcn/utils.py` | Utility functions |
| `gcn/layers.py` | GCN layers |

### Data Loading Pipeline

```
Raw Data (pickled files)
         ↓
  load_data()
         ↓
Sparse matrices & arrays
         ↓
  preprocess_features()
  preprocess_adj()
         ↓
Normalized sparse tensors
         ↓
TensorFlow placeholders
         ↓
Training
```

### Training Loop

```
Epoch 1-200:
  ├─ Forward pass on training data
  ├─ Compute cross-entropy loss + L2 regularization
  ├─ Backward pass (gradient computation)
  ├─ Parameter update (Adam optimizer)
  ├─ Validate on validation set
  ├─ Check early stopping criterion
  └─ Log metrics

Final: Evaluate on test set
```

---

## Paper Comparison

### Claimed Performance (Paper)

- **Cora:** 81.5% ± 0.5%
- **Citeseer:** 70.3% ± 0.7%
- **Pubmed:** 79.0% ± 0.3%

### Expected Performance (This Implementation)

Should match paper results within ±1% when using:
- Same hyperparameters
- Same random seed
- TensorFlow 1.15.4 (or compatible)

---

## Advanced Usage

### Modifying Hyperparameters

```python
from config import GCNConfig, ModelConfig, TrainingConfig

config = GCNConfig(
    model=ModelConfig(
        hidden1=32,           # Increase hidden units
        dropout=0.6,          # Increase dropout
        weight_decay=1e-3,    # More regularization
    ),
    training=TrainingConfig(
        learning_rate=0.005,  # Lower learning rate
        epochs=300,           # More epochs
    )
)

config.validate()
# ... run experiment
```

### Using Chebyshev Polynomial Approximation

```python
from config import GCNConfig, ModelConfig

config = GCNConfig(
    model=ModelConfig(model_type='gcn_cheby'),
    graph=GraphConfig(max_degree=5),  # Chebyshev degree
)
```

### Custom Dataset

For custom node classification datasets:

```python
from data_loader import GCNDataLoader

loader = GCNDataLoader()
# Implement custom load function
# Should return: (adj, features, y_train, y_val, y_test,
#                 train_mask, val_mask, test_mask)
```

---

## References

1. **Paper:** Kipf & Welling (2017)
   - [ArXiv](https://arxiv.org/abs/1609.02907)
   - [Official Implementation](https://github.com/tkipf/gcn)

2. **Datasets:** Planetoid Project
   - [GitHub](https://github.com/kimiyoung/planetoid)
   - [Paper (Yang et al. 2016)](https://arxiv.org/abs/1603.08861)

3. **Spectral Methods:**
   - Defferrard et al. (2016) - [CNN on Graphs](https://arxiv.org/abs/1606.09375)

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

---

**Last Updated:** 2024-02-06  
**Status:** Fully Tested & Reproducible ✓
