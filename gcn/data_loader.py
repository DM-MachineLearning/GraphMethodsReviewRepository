"""
GCN Data Loader
==============
Flexible data loading and preprocessing for Graph Convolutional Networks.

Supports:
- Citation networks (Cora, Citeseer, Pubmed) from Kipf's implementation
- Custom formats (npz, npy, csv)
- Automatic feature and adjacency matrix preprocessing
"""

import os
import sys
import json
import logging
from typing import Dict, Tuple, Optional, Union
import numpy as np
import scipy.sparse as sp

# Add gcn package to path
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


class GCNDataLoader:
    """Load and preprocess data for GCN."""
    
    def __init__(self, data_path: str = './gcn/data', normalize_features: bool = True):
        """Initialize data loader.
        
        Args:
            data_path: Path to data directory
            normalize_features: Whether to normalize features
        """
        self.data_path = data_path
        self.normalize_features = normalize_features
    
    def load_citation_network(self, dataset_name: str = 'cora') -> Tuple:
        """Load citation network from Kipf's implementation format.
        
        Args:
            dataset_name: 'cora', 'citeseer', or 'pubmed'
        
        Returns:
            Tuple of (adj, features, y_train, y_val, y_test, 
                     train_mask, val_mask, test_mask)
        """
        from gcn.utils import load_data
        return load_data(dataset_name)
    
    def load_npz(self, filepath: str) -> Tuple:
        """Load data from NPZ file.
        
        Expected keys: 'adj', 'features', 'labels'
        """
        data = np.load(filepath, allow_pickle=True)
        adj = sp.csr_matrix(data['adj'])
        features = sp.csr_matrix(data['features']) if sp.issparse(data['features']) else data['features']
        labels = data['labels']
        
        return adj, features, labels
    
    def load_csv(self, adj_file: str, features_file: str, labels_file: str) -> Tuple:
        """Load data from CSV files."""
        adj = np.loadtxt(adj_file, delimiter=',')
        features = np.loadtxt(features_file, delimiter=',')
        labels = np.loadtxt(labels_file, delimiter=',', dtype=int)
        
        adj = sp.csr_matrix(adj)
        features = sp.csr_matrix(features)
        
        return adj, features, labels


class DataConfig:
    """Configuration for dataset paths and parameters."""
    
    DATASETS = {
        'cora': {
            'path': './gcn/data',
            'splits': {'train': 0.6, 'val': 0.2, 'test': 0.2},
            'features': 1433,
            'classes': 7,
        },
        'citeseer': {
            'path': './gcn/data',
            'splits': {'train': 0.6, 'val': 0.2, 'test': 0.2},
            'features': 3703,
            'classes': 6,
        },
        'pubmed': {
            'path': './gcn/data',
            'splits': {'train': 0.6, 'val': 0.2, 'test': 0.2},
            'features': 500,
            'classes': 3,
        },
    }
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict:
        """Get dataset information."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.DATASETS[dataset_name]


def save_dataset_info(output_file: str, config_dict: Dict):
    """Save dataset information to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


if __name__ == '__main__':
    # Example usage
    print("GCN Data Loader Module")
    print("Supports loading citation networks and custom data formats")
    
    # Print available datasets
    print("\nAvailable Datasets:")
    for dataset_name, info in DataConfig.DATASETS.items():
        print(f"  - {dataset_name}: {info['features']} features, {info['classes']} classes")
