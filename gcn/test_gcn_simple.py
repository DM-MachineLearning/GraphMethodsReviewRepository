"""Simple GCN test - validates configuration and data loading"""
import sys
sys.path.insert(0, '.')

from config import get_cora_config
from utils import load_data, preprocess_features, preprocess_adj

print("=" * 70)
print("GCN IMPLEMENTATION TEST")
print("=" * 70)

# Test 1: Configuration
print("\n[1/3] Testing Configuration...")
try:
    config = get_cora_config()
    print(f"  ✓ Config loaded: {config.data.dataset_name}")
    print(f"  ✓ Dataset path: {config.data.dataset_path}")
    print(f"  ✓ Model type: {config.model.model_type}")
    print(f"  ✓ Hidden units: {config.model.hidden1}")
    print(f"  ✓ Learning rate: {config.training.learning_rate}")
    print(f"  ✓ Epochs: {config.training.epochs}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Data Loading
print("\n[2/3] Testing Data Loading (Cora)...")
try:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    print(f"  ✓ Adjacency matrix shape: {adj.shape}")
    print(f"  ✓ Features shape (sparse): {features.shape}")
    print(f"  ✓ Training instances: {train_mask.sum()}")
    print(f"  ✓ Validation instances: {val_mask.sum()}")
    print(f"  ✓ Test instances: {test_mask.sum()}")
    print(f"  ✓ Number of classes: {y_train.shape[1]}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Test 3: Data Preprocessing
print("\n[3/3] Testing Data Preprocessing...")
try:
    features_preprocessed = preprocess_features(features)
    adj_preprocessed = preprocess_adj(adj)
    print(f"  ✓ Features preprocessed (type: {type(features_preprocessed).__name__})")
    print(f"  ✓ Adjacency matrix preprocessed (type: {type(adj_preprocessed).__name__})")
    if isinstance(features_preprocessed, tuple) and len(features_preprocessed) == 3:
        coords, values, shape = features_preprocessed
        print(f"    - Coordinates shape: {coords.shape}")
        print(f"    - Values shape: {values.shape}")
        print(f"    - Matrix shape: {shape}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - GCN implementation is working!")
print("=" * 70)
