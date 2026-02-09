# GAT Quick Reference

Fast lookup for GAT configuration parameters, common patterns, and troubleshooting.

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Cora dataset
python -c "
import sys
sys.path.insert(0, '.')
from gat.run_experiment import GATExperiment
from gat.config_example_cora import get_config
exp = GATExperiment(get_config())
results = exp.run()
print(f'Test Accuracy: {results[\"test_accuracy\"]:.4f}')
"

# Run custom experiment
python -c "
import sys
sys.path.insert(0, '.')
from gat.config import GATConfig, DataConfig, ModelConfig, TrainingConfig, OutputConfig
config = GATConfig(
    data=DataConfig(dataset_name='citeseer'),
    model=ModelConfig(hid_units=[16], n_heads=[8, 1]),
    training=TrainingConfig(epochs=50000, patience=100)
)
from gat.run_experiment import GATExperiment
exp = GATExperiment(config)
results = exp.run()
"
```

---

## Configuration Parameter Reference

### DataConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | "cora" | Dataset: "cora", "citeseer", or "pubmed" |
| `batch_size` | int | 1 | Batch size (typically 1 for GAT) |
| `nhood` | int | 1 | Neighborhood expansion for adjacency matrix |
| `data_dir` | str | "data" | Directory containing dataset files |

**Example:**
```python
from gat.config import DataConfig

# Cora dataset
data = DataConfig(dataset_name="cora", batch_size=1)

# PubMed with custom data directory
data = DataConfig(dataset_name="pubmed", data_dir="/path/to/data")
```

---

### ModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hid_units` | List[int] | [8] | Hidden units per layer |
| `n_heads` | List[int] | [8, 1] | Attention heads per layer |
| `activation` | str | "elu" | Activation function: "elu", "relu", "sigmoid" |
| `residual` | bool | False | Enable residual connections |
| `sparse` | bool | False | Use sparse operations (for large graphs) |
| `dropout_attn` | float | 0.6 | Attention dropout (0.0-1.0) |
| `dropout_ffd` | float | 0.6 | Feed-forward dropout (0.0-1.0) |

**Common Configurations:**

```python
from gat.config import ModelConfig

# Standard GAT (default)
model = ModelConfig(hid_units=[8], n_heads=[8, 1])

# Deeper model
model = ModelConfig(hid_units=[8, 8], n_heads=[8, 8, 1])

# Smaller model (lower GPU memory)
model = ModelConfig(hid_units=[4], n_heads=[4, 1], dropout_attn=0.5)

# Larger model (better accuracy, more memory)
model = ModelConfig(hid_units=[16, 16], n_heads=[8, 8, 1])

# Sparse mode for large graphs
model = ModelConfig(sparse=True, hid_units=[8], n_heads=[8, 1])

# With residual connections
model = ModelConfig(hid_units=[8], n_heads=[8, 1], residual=True)
```

---

### TrainingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100000 | Maximum training epochs |
| `patience` | int | 100 | Early stopping patience |
| `learning_rate` | float | 0.005 | Optimizer learning rate |
| `l2_coef` | float | 0.0005 | L2 regularization coefficient |
| `attn_drop` | float | 0.6 | Attention dropout rate |
| `ffd_drop` | float | 0.6 | Feed-forward dropout rate |

**Common Configurations:**

```python
from gat.config import TrainingConfig

# Default (balanced)
training = TrainingConfig()

# Fast training (fewer epochs)
training = TrainingConfig(epochs=10000, patience=50)

# Slow training (more regularization, better accuracy)
training = TrainingConfig(
    epochs=200000, 
    patience=200,
    learning_rate=0.001,
    l2_coef=0.001,
    attn_drop=0.8,
    ffd_drop=0.8
)

# Low regularization (potential overfitting)
training = TrainingConfig(
    learning_rate=0.01,
    l2_coef=0.0,
    attn_drop=0.3,
    ffd_drop=0.3
)
```

---

### OutputConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | str | "checkpoints" | Directory for model checkpoints |
| `log_dir` | str | "logs" | Directory for training logs |
| `result_dir` | str | "results" | Directory for results JSON |

**Example:**
```python
from gat.config import OutputConfig

output = OutputConfig(
    checkpoint_dir="./experiments/checkpoints",
    log_dir="./experiments/logs",
    result_dir="./experiments/results"
)
```

---

## Complete Configuration Examples

### Example 1: Cora Dataset (Balanced)
```python
from gat.config import GATConfig, DataConfig, ModelConfig, TrainingConfig, OutputConfig

config = GATConfig(
    data=DataConfig(dataset_name="cora", batch_size=1, nhood=1),
    model=ModelConfig(hid_units=[8], n_heads=[8, 1], dropout_attn=0.6, dropout_ffd=0.6),
    training=TrainingConfig(epochs=100000, patience=100, learning_rate=0.005, l2_coef=0.0005),
    output=OutputConfig()
)

# Expected result: ~83% test accuracy
```

### Example 2: CiteSeer Dataset
```python
config = GATConfig(
    data=DataConfig(dataset_name="citeseer"),
    model=ModelConfig(hid_units=[8], n_heads=[8, 1]),
    training=TrainingConfig(epochs=100000, patience=200, learning_rate=0.01)
)

# Expected result: ~72% test accuracy
```

### Example 3: PubMed (Sparse Mode)
```python
config = GATConfig(
    data=DataConfig(dataset_name="pubmed", batch_size=1),
    model=ModelConfig(
        hid_units=[8], 
        n_heads=[8, 1],
        sparse=True,  # Use sparse operations
        dropout_attn=0.1,
        dropout_ffd=0.1
    ),
    training=TrainingConfig(epochs=50000, patience=100, learning_rate=0.01)
)

# Note: Sparse mode requires batch_size=1
```

### Example 4: Small Model (Low Memory)
```python
config = GATConfig(
    data=DataConfig(dataset_name="cora"),
    model=ModelConfig(
        hid_units=[4],      # 4 instead of 8
        n_heads=[4, 1],     # 4 instead of 8
        dropout_attn=0.3,
        dropout_ffd=0.3
    ),
    training=TrainingConfig(learning_rate=0.01)
)

# Reduced memory usage, slightly lower accuracy
```

### Example 5: Large Model (High Accuracy)
```python
config = GATConfig(
    data=DataConfig(dataset_name="cora"),
    model=ModelConfig(
        hid_units=[16, 16],  # Deeper
        n_heads=[8, 8, 1],   # More heads
        dropout_attn=0.6,
        dropout_ffd=0.6
    ),
    training=TrainingConfig(
        epochs=200000, 
        patience=200, 
        learning_rate=0.001,
        l2_coef=0.001
    )
)

# Higher accuracy, longer training, more memory
```

---

## Running Experiments

### Method 1: Using Config Files

```python
import sys
sys.path.insert(0, '/path/to/repo')

from gat.run_experiment import GATExperiment
from gat.config_example_cora import get_config

config = get_config()
experiment = GATExperiment(config)
results = experiment.run()

print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"Test Loss: {results['test_loss']:.4f}")
```

### Method 2: Using GATConfig Directly

```python
from gat.config import GATConfig
from gat.run_experiment import GATExperiment

config = GATConfig(
    data__dataset_name="cora",
    model__hid_units=[8],
    training__epochs=100000
)

experiment = GATExperiment(config)
results = experiment.run()
```

### Method 3: Programmatic Loop

```python
import random
import numpy as np
import tensorflow as tf
from gat.config import GATConfig
from gat.run_experiment import GATExperiment

accuracies = []

for seed in range(5):
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # Run experiment
    config = GATConfig()
    experiment = GATExperiment(config)
    results = experiment.run()
    
    accuracies.append(results['test_accuracy'])
    print(f"Seed {seed}: {results['test_accuracy']:.4f}")

print(f"\nMean: {np.mean(accuracies):.4f}")
print(f"Std:  {np.std(accuracies):.4f}")
```

---

## Dataset Information

### Cora
- **Nodes:** 2,708
- **Edges:** 5,429
- **Features:** 1,433 (bag-of-words)
- **Classes:** 7 (ML research areas)
- **Train/Val/Test:** 140 / 500 / 1,000
- **Recommended config:** `hid_units=[8], n_heads=[8,1], lr=0.005`

### CiteSeer
- **Nodes:** 3,312
- **Edges:** 4,732
- **Features:** 3,703 (bag-of-words)
- **Classes:** 6 (ML research areas)
- **Train/Val/Test:** 120 / 500 / 1,000
- **Recommended config:** `hid_units=[8], n_heads=[8,1], lr=0.01`

### PubMed
- **Nodes:** 19,717
- **Edges:** 44,338
- **Features:** 500
- **Classes:** 3 (medical topics)
- **Train/Val/Test:** 60 / 500 / 19,157
- **Recommended config:** `hid_units=[8], n_heads=[8,1], sparse=True, lr=0.01`

---

## Hyperparameter Tuning Guide

### If test accuracy is too low:

1. **Increase model capacity:**
   ```python
   hid_units=[16]          # Increase from [8] to [16]
   n_heads=[8, 1]          # Already good
   ```

2. **Decrease regularization:**
   ```python
   attn_drop=0.3           # Decrease from 0.6
   ffd_drop=0.3            # Decrease from 0.6
   l2_coef=0.0             # Decrease from 0.0005
   ```

3. **Increase training time:**
   ```python
   epochs=200000           # Increase from 100000
   patience=200            # Increase from 100
   ```

### If training is unstable (loss spikes):

1. **Decrease learning rate:**
   ```python
   learning_rate=0.001     # Decrease from 0.005
   ```

2. **Increase regularization:**
   ```python
   attn_drop=0.8           # Increase from 0.6
   ffd_drop=0.8            # Increase from 0.6
   l2_coef=0.001           # Increase from 0.0005
   ```

### If running out of memory:

1. **Reduce model size:**
   ```python
   hid_units=[4]           # Reduce from [8]
   n_heads=[4, 1]          # Reduce from [8, 1]
   ```

2. **Use sparse mode:**
   ```python
   sparse=True
   ```

3. **Reduce batch size (already at 1):**
   Limited by GAT architecture

---

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| ImportError: No module named 'gat' | `sys.path.insert(0, '/path/to/repo')` |
| CUDA out of memory | Reduce `hid_units`, increase `attn_drop` |
| Loss NaN | Lower `learning_rate` to 0.001 |
| Accuracy stuck at ~20% | Increase `epochs` or `hid_units` |
| Very slow training | Use GPU, reduce `epochs` for testing |

---

## Expected Performance

| Dataset | Config | Test Accuracy |
|---------|--------|---------------|
| Cora | Default | 83.0 ± 0.7% |
| CiteSeer | Default | 72.5 ± 0.7% |
| PubMed | Sparse | 79.0 ± 0.3% |

---

## File Structure

```
gat/
├── __init__.py                    # Package initialization
├── config.py                      # Configuration dataclasses
├── data_loader.py                 # Data loading utilities
├── run_experiment.py              # Training pipeline
├── config_example_cora.py         # Example Cora config
├── README_REPRODUCIBLE.md         # Comprehensive guide
├── QUICK_REFERENCE.md             # This file
├── models/
│   ├── __init__.py
│   ├── gat.py                     # GAT model implementation
│   ├── base_gattn.py              # Base attention class
│   └── sp_gat.py                  # Sparse GAT
├── utils/
│   ├── __init__.py
│   ├── layers.py                  # Attention layers
│   ├── process.py                 # Data preprocessing
│   └── process_ppi.py             # PPI-specific processing
├── data/
│   ├── ind.cora.x                 # Cora features
│   ├── ind.cora.y                 # Cora labels
│   └── ... (other datasets)
└── checkpoints/
    └── best_model/                # Best model checkpoint
```

---

## Common Patterns

### Pattern 1: Grid Search Over Learning Rates
```python
from gat.run_experiment import GATExperiment
from gat.config import GATConfig

learning_rates = [0.001, 0.005, 0.01, 0.02]
best_acc = 0

for lr in learning_rates:
    config = GATConfig(training__learning_rate=lr)
    exp = GATExperiment(config)
    results = exp.run()
    if results['test_accuracy'] > best_acc:
        best_acc = results['test_accuracy']
        best_lr = lr

print(f"Best LR: {best_lr}, Accuracy: {best_acc:.4f}")
```

### Pattern 2: Ablation Study
```python
# Compare with/without attention dropout
configs = [
    GATConfig(model__dropout_attn=0.0),
    GATConfig(model__dropout_attn=0.3),
    GATConfig(model__dropout_attn=0.6),
    GATConfig(model__dropout_attn=0.9),
]

for i, config in enumerate(configs):
    exp = GATExperiment(config)
    results = exp.run()
    print(f"Dropout {config.model.dropout_attn}: {results['test_accuracy']:.4f}")
```

### Pattern 3: Save Best Model for Later Use
```python
config = GATConfig(
    output__checkpoint_dir="./my_model"
)

exp = GATExperiment(config)
results = exp.run()

# Later: Load and use model
# (See run_experiment.py for checkpoint loading)
```

---

## Version Information

- **GAT Paper:** Veličković et al., ICLR 2018
- **Original Repository:** https://github.com/PetarV-/GAT
- **Implementation Version:** 1.0.0
- **TensorFlow Requirement:** >= 1.6 (tested on 1.15, 2.x)
- **Python Requirement:** >= 3.5

---

## Additional Resources

- **Papers:**
  - GAT: https://arxiv.org/abs/1710.10903
  - GCN: https://arxiv.org/abs/1609.02907
  - Attention: https://arxiv.org/abs/1706.03762

- **Documentation:**
  - TensorFlow: https://www.tensorflow.org/
  - NumPy: https://numpy.org/doc/
  - NetworkX: https://networkx.org/

---

**Last Updated:** February 2025
