"""
Graph Convolutional Networks - Configuration System
=====================================
Centralized configuration for GCN with reproducibility and validation.

Based on: Kipf & Welling (2017) - Semi-Supervised Classification with GCN
Paper: https://arxiv.org/abs/1609.02907

Configuration Classes:
- DataConfig: Dataset selection, paths, and loading parameters
- GraphConfig: Graph preprocessing and structure parameters  
- ModelConfig: GCN architecture (hidden layers, filters, etc.)
- TrainingConfig: Learning parameters (epochs, learning rate, regularization)
- OutputConfig: Output paths and logging
- ValidationConfig: Configuration validation and compatibility checking
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Dataset
    dataset_name: str = 'cora'  # 'cora', 'citeseer', 'pubmed'
    dataset_path: str = './data'
    
    # Data splits (for custom datasets)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Feature preprocessing
    normalize_features: bool = True
    sparse_features: bool = True
    
    def validate(self):
        """Validate data configuration."""
        valid_datasets = ['cora', 'citeseer', 'pubmed']
        if self.dataset_name not in valid_datasets:
            raise ValueError(f"Dataset must be one of {valid_datasets}, got {self.dataset_name}")
        
        assert 0 < self.train_ratio <= 1, "train_ratio must be in (0, 1]"
        assert 0 <= self.val_ratio < 1, "val_ratio must be in [0, 1)"
        assert 0 <= self.test_ratio < 1, "test_ratio must be in [0, 1)"
        
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert 0.99 <= total <= 1.01, f"Ratios must sum to 1, got {total}"


@dataclass
class GraphConfig:
    """Graph preprocessing and structure configuration."""
    
    # Adjacency matrix preprocessing
    add_self_loops: bool = True
    normalize_adj: bool = True
    symmetric_normalize: bool = True
    
    # Graph structure
    max_degree: int = 3  # For Chebyshev polynomial approximation
    
    def validate(self):
        """Validate graph configuration."""
        assert self.max_degree >= 1, "max_degree must be >= 1"


@dataclass
class ModelConfig:
    """GCN model architecture configuration."""
    
    # Model type
    model_type: str = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
    
    # Architecture
    hidden1: int = 16  # Hidden layer size (from paper)
    hidden2: Optional[int] = None  # Optional second hidden layer
    dropout: float = 0.5  # Dropout rate
    
    # Regularization
    weight_decay: float = 5e-4  # L2 regularization
    
    # Activation functions
    activation: str = 'relu'  # 'relu', 'elu', 'sigmoid', 'tanh'
    
    def validate(self):
        """Validate model configuration."""
        valid_models = ['gcn', 'gcn_cheby', 'dense']
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}, got {self.model_type}")
        
        assert self.hidden1 > 0, "hidden1 must be > 0"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.weight_decay >= 0, "weight_decay must be >= 0"
        
        valid_activations = ['relu', 'elu', 'sigmoid', 'tanh']
        if self.activation not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}, got {self.activation}")


@dataclass
class TrainingConfig:
    """Training parameters configuration."""
    
    # Training schedule
    epochs: int = 200  # From paper
    learning_rate: float = 0.01  # From paper
    
    # Early stopping
    early_stopping: int = 10  # Patience for early stopping
    early_stopping_metric: str = 'val_loss'  # 'val_loss' or 'val_acc'
    
    # Batch processing
    batch_size: int = None  # None = full batch (GCN default)
    
    # Optimizer
    optimizer: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    
    # Learning rate schedule
    lr_decay: bool = False
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 10
    
    # Random seed
    seed: int = 123
    
    def validate(self):
        """Validate training configuration."""
        assert self.epochs > 0, "epochs must be > 0"
        assert self.learning_rate > 0, "learning_rate must be > 0"
        assert self.early_stopping > 0, "early_stopping must be > 0"
        
        valid_optimizers = ['adam', 'sgd', 'rmsprop']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}, got {self.optimizer}")
        
        if self.batch_size is not None:
            assert self.batch_size > 0, "batch_size must be > 0"


@dataclass
class OutputConfig:
    """Output paths and logging configuration."""
    
    # Output directories
    output_dir: str = './outputs'
    checkpoints_dir: str = './outputs/checkpoints'
    logs_dir: str = './outputs/logs'
    results_dir: str = './outputs/results'
    
    # Output files
    save_model: bool = True
    save_results: bool = True
    save_config: bool = True
    
    # Logging
    verbose: bool = True
    log_every_n_steps: int = 10
    
    def validate(self):
        """Validate output configuration."""
        assert self.log_every_n_steps > 0, "log_every_n_steps must be > 0"


@dataclass
class RegularizationConfig:
    """Additional regularization techniques."""
    
    # Dropout variants
    spatial_dropout: bool = False
    
    # Batch normalization
    batch_norm: bool = False
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    def validate(self):
        """Validate regularization configuration."""
        assert 0 <= self.label_smoothing < 1, "label_smoothing must be in [0, 1)"


class GCNConfig:
    """
    Complete GCN configuration combining all sub-configurations.
    
    Provides:
    - Unified parameter management
    - Cross-component validation
    - Serialization/deserialization
    - Pretty printing
    """
    
    def __init__(
        self,
        data: DataConfig = None,
        graph: GraphConfig = None,
        model: ModelConfig = None,
        training: TrainingConfig = None,
        output: OutputConfig = None,
        regularization: RegularizationConfig = None,
    ):
        """Initialize GCN configuration with all sub-configs."""
        self.data = data or DataConfig()
        self.graph = graph or GraphConfig()
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.output = output or OutputConfig()
        self.regularization = regularization or RegularizationConfig()
    
    def validate(self):
        """Validate entire configuration."""
        print("[CONFIG] Validating configuration...")
        
        # Validate each sub-config
        self.data.validate()
        print("  ✓ Data config valid")
        
        self.graph.validate()
        print("  ✓ Graph config valid")
        
        self.model.validate()
        print("  ✓ Model config valid")
        
        self.training.validate()
        print("  ✓ Training config valid")
        
        self.output.validate()
        print("  ✓ Output config valid")
        
        self.regularization.validate()
        print("  ✓ Regularization config valid")
        
        # Cross-component validation
        self._validate_compatibility()
        print("  ✓ Cross-component compatibility verified")
        print("[CONFIG] Configuration valid! ✓\n")
    
    def _validate_compatibility(self):
        """Check compatibility between components."""
        # Check dataset path exists
        if not os.path.exists(self.data.dataset_path):
            print(f"  ⚠ Warning: Dataset path does not exist: {self.data.dataset_path}")
        
        # Check model-graph compatibility
        if self.model.model_type == 'gcn_cheby' and self.graph.max_degree < 1:
            raise ValueError("Chebyshev GCN requires max_degree >= 1")
        
        # Check early stopping metric validity
        valid_metrics = ['val_loss', 'val_acc']
        if self.training.early_stopping_metric not in valid_metrics:
            raise ValueError(f"Early stopping metric must be one of {valid_metrics}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'graph': asdict(self.graph),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'output': asdict(self.output),
            'regularization': asdict(self.regularization),
        }
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def from_json(self, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.data = DataConfig(**config_dict['data'])
        self.graph = GraphConfig(**config_dict['graph'])
        self.model = ModelConfig(**config_dict['model'])
        self.training = TrainingConfig(**config_dict['training'])
        self.output = OutputConfig(**config_dict['output'])
        self.regularization = RegularizationConfig(**config_dict['regularization'])
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2)


# Default configurations for different datasets
def get_cora_config() -> GCNConfig:
    """Get recommended configuration for Cora dataset."""
    return GCNConfig(
        data=DataConfig(dataset_name='cora', dataset_path='./gcn/data'),
        model=ModelConfig(model_type='gcn', hidden1=16),
        training=TrainingConfig(epochs=200, learning_rate=0.01, early_stopping=10),
    )


def get_citeseer_config() -> GCNConfig:
    """Get recommended configuration for Citeseer dataset."""
    return GCNConfig(
        data=DataConfig(dataset_name='citeseer', dataset_path='./gcn/data'),
        model=ModelConfig(model_type='gcn', hidden1=16),
        training=TrainingConfig(epochs=200, learning_rate=0.01, early_stopping=10),
    )


def get_pubmed_config() -> GCNConfig:
    """Get recommended configuration for Pubmed dataset."""
    return GCNConfig(
        data=DataConfig(dataset_name='pubmed', dataset_path='./gcn/data'),
        model=ModelConfig(model_type='gcn', hidden1=16),
        training=TrainingConfig(epochs=200, learning_rate=0.01, early_stopping=10),
    )


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("GCN Configuration System - Example")
    print("=" * 60)
    
    # Create default config
    config = get_cora_config()
    config.validate()
    
    print("Configuration:")
    print(config)
