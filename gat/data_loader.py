"""
Data loading utilities for GAT.
"""

import numpy as np
from typing import Dict, Any

from .utils import process
from .config import DataConfig


def load_citation_data(config: DataConfig) -> Dict[str, Any]:
    """Load citation dataset and build bias/feature tensors."""
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(
        config.dataset_name
    )

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    adj = adj.todense()

    # Add batch dimension
    features = features[np.newaxis]
    adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = process.adj_to_bias(adj, [nb_nodes], nhood=config.nhood)

    return {
        "adj": adj,
        "features": features,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "biases": biases,
        "nb_nodes": nb_nodes,
        "ft_size": ft_size,
        "nb_classes": nb_classes,
    }
