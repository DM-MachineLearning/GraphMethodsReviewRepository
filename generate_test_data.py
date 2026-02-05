#!/usr/bin/env python3
"""
Generate sample MNIST data for testing.
Creates grid graphs from random 28x28 images.
"""

import numpy as np
from pathlib import Path

def generate_mnist_sample(n_samples=100):
    """Generate sample MNIST-like data."""
    
    print("Generating sample MNIST data...")
    print(f"  - Samples: {n_samples}")
    print(f"  - Image size: 28×28 (784 features)")
    print(f"  - Classes: 10")
    
    # Generate random 28x28 images flattened to 784 features
    X = np.random.rand(n_samples, 784).astype(np.float32)
    
    # Generate random labels (0-9)
    Y = np.random.randint(0, 10, n_samples).astype(np.int32)
    
    # Ensure all classes represented
    for i in range(10):
        if i not in Y:
            Y[i % n_samples] = i
    
    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Save data
    features_file = data_dir / 'features.npz'
    labels_file = data_dir / 'labels.npy'
    
    np.savez(features_file, features=X)
    np.save(labels_file, Y)
    
    print(f"\n✓ Data saved:")
    print(f"  - Features: {features_file} ({X.shape})")
    print(f"  - Labels: {labels_file} ({Y.shape})")
    print(f"  - Classes: {len(np.unique(Y))}")
    
    return X, Y

if __name__ == '__main__':
    X, Y = generate_mnist_sample(n_samples=200)
    print(f"\nData shapes:")
    print(f"  X: {X.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  Unique classes: {np.unique(Y)}")
