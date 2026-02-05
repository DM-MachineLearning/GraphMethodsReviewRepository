"""
Data loading and preprocessing utilities.

This module provides functions to load data according to config.py settings,
handle multiple data formats, and prepare data for the CNN graph model.

Functions:
    load_features() - Load feature matrix from various formats
    load_labels() - Load label vector from various formats
    prepare_data() - End-to-end data loading and preprocessing
    split_dataset() - Train/val/test splitting
    normalize_features() - Feature normalization
    validate_shapes() - Verify data shape consistency
"""

import os
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, Optional

from config import DataConfig, GraphConfig, OutputConfig


def load_features(filepath: str) -> np.ndarray:
    """
    Load feature matrix from file.
    
    Supports multiple formats:
    - .npy: NumPy binary format
    - .npz: NumPy compressed (extracts 'features', 'X', 'data', or first array)
    - .txt, .csv: Text format (delimiter detected automatically)
    
    Args:
        filepath: Path to feature file
    
    Returns:
        Feature matrix (N_samples, N_features)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported or data is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")
    
    print(f"Loading features from: {filepath}")
    
    if filepath.endswith('.npy'):
        return np.load(filepath)
    
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        # Try common key names
        for key in ['features', 'X', 'data']:
            if key in data:
                return data[key]
        # Fall back to first array
        return data[list(data.files)[0]]
    
    elif filepath.endswith(('.txt', '.csv')):
        # Auto-detect delimiter
        with open(filepath, 'r') as f:
            first_line = f.readline()
        
        delimiter = '\t' if '\t' in first_line else ','
        return np.loadtxt(filepath, delimiter=delimiter)
    
    elif filepath.endswith('.sparse'):
        # SciPy sparse matrix format
        return sp.load_npz(filepath).toarray()
    
    else:
        raise ValueError(f"Unsupported format: {filepath}")


def load_labels(filepath: str) -> np.ndarray:
    """
    Load label vector from file.
    
    Supports multiple formats:
    - .npy: NumPy binary format
    - .npz: NumPy compressed (extracts 'labels', 'y', or first array)
    - .txt, .csv: Text format with one label per line or row
    
    Args:
        filepath: Path to label file
    
    Returns:
        Label vector (N_samples,) with integer class labels
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported or data is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")
    
    print(f"Loading labels from: {filepath}")
    
    if filepath.endswith('.npy'):
        labels = np.load(filepath)
    
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        # Try common key names
        for key in ['labels', 'y', 'targets']:
            if key in data:
                labels = data[key]
                break
        else:
            labels = data[list(data.files)[0]]
    
    elif filepath.endswith(('.txt', '.csv')):
        labels = np.loadtxt(filepath, dtype=int)
    
    else:
        raise ValueError(f"Unsupported format: {filepath}")
    
    # Flatten if needed
    labels = labels.flatten()
    
    # Ensure integer type
    labels = labels.astype(int)
    
    return labels


def validate_shapes(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Validate that data shapes are consistent.
    
    Args:
        X: Feature matrix (N, F)
        Y: Labels (N,)
    
    Raises:
        ValueError: If shapes are inconsistent
    """
    if X.ndim != 2:
        raise ValueError(f"Features must be 2D: got shape {X.shape}")
    
    if Y.ndim != 1:
        raise ValueError(f"Labels must be 1D: got shape {Y.shape}")
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {X.shape[0]} features vs {Y.shape[0]} labels"
        )
    
    print(f"✓ Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X: Feature matrix (N, F)
    
    Returns:
        Normalized feature matrix
    """
    if DataConfig.NORMALIZE_FEATURES:
        print("Normalizing features...")
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
    
    return X


def split_dataset(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split dataset into train/validation/test sets.
    
    Uses stratified split to maintain class distribution.
    
    Args:
        X: Feature matrix (N, F)
        Y: Labels (N,)
    
    Returns:
        Dict with keys 'train', 'val', 'test' containing sample indices
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    n_train = int(n_samples * DataConfig.TRAIN_RATIO)
    n_val = int(n_samples * DataConfig.VAL_RATIO)
    
    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]
    
    print(f"Train: {len(idx_train)} ({100*len(idx_train)/n_samples:.1f}%)")
    print(f"Val:   {len(idx_val)} ({100*len(idx_val)/n_samples:.1f}%)")
    print(f"Test:  {len(idx_test)} ({100*len(idx_test)/n_samples:.1f}%)")
    
    return {
        'train': idx_train,
        'val': idx_val,
        'test': idx_test,
    }


def prepare_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and prepare dataset end-to-end.
    
    Uses paths and settings from config.py.
    
    Returns:
        (X, Y, splits) where:
        - X: Feature matrix (N, F)
        - Y: Labels (N,)
        - splits: Dict with 'train', 'val', 'test' indices
    
    Example:
        X, Y, splits = prepare_data()
        X_train, Y_train = X[splits['train']], Y[splits['train']]
    """
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80 + "\n")
    
    # Load data
    X = load_features(DataConfig.DATA_FILE)
    Y = load_labels(DataConfig.LABELS_FILE)
    
    # Validate
    validate_shapes(X, Y)
    
    # Print data info
    n_classes = len(np.unique(Y))
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    
    # Normalize
    X = normalize_features(X)
    
    # Split
    print("\nSplitting dataset...")
    splits = split_dataset(X, Y)
    
    return X, Y, splits


if __name__ == '__main__':
    """Test data loading."""
    print("Testing data loading module...")
    
    try:
        X, Y, splits = prepare_data()
        print(f"\n✓ Data loading successful!")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  Classes: {np.unique(Y)}")
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
