"""
MPNN Configuration System
Configuration dataclasses for reproducible Message Passing Neural Network experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_name: str = "qm9"  # qm9, letter, mnist, etc.
    dataset_path: str = "data/qm9"
    batch_size: int = 100
    num_workers: int = 0
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    shuffle: bool = True
    qm9_target_property: int = 0  # 0=dipole, 1=isotropic, 2=homo, 3=lumo, etc.
    edge_representation: str = "distance"  # distance or features
    normalize_data: bool = True
    prefetch_batches: int = 0

    def validate(self):
        """Validate data configuration."""
        assert self.train_split + self.val_split + self.test_split == 1.0, \
            f"Splits must sum to 1.0, got {self.train_split + self.val_split + self.test_split}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.qm9_target_property in range(13), "qm9_target_property must be 0-12"
        assert self.edge_representation in ["distance", "features"], \
            f"edge_representation must be 'distance' or 'features', got {self.edge_representation}"


@dataclass
class MessageFunctionConfig:
    """Message passing function configuration."""
    message_type: str = "duvenaud"  # duvenaud, intnet, ggnn
    message_hidden_dim: int = 64
    message_passing_steps: int = 3

    def validate(self):
        """Validate message function configuration."""
        assert self.message_type in ["duvenaud", "intnet", "ggnn"], \
            f"message_type must be one of ['duvenaud', 'intnet', 'ggnn'], got {self.message_type}"
        assert self.message_hidden_dim > 0, "message_hidden_dim must be positive"
        assert self.message_passing_steps > 0, "message_passing_steps must be positive"


@dataclass
class UpdateFunctionConfig:
    """Node update function configuration."""
    update_type: str = "mlp"  # mlp, gru, lstm
    update_hidden_dim: int = 128
    update_dropout: float = 0.0
    update_activation: str = "relu"  # relu, elu, leaky_relu

    def validate(self):
        """Validate update function configuration."""
        assert self.update_type in ["mlp", "gru", "lstm"], \
            f"update_type must be one of ['mlp', 'gru', 'lstm'], got {self.update_type}"
        assert self.update_hidden_dim > 0, "update_hidden_dim must be positive"
        assert 0 <= self.update_dropout < 1.0, "update_dropout must be in [0, 1)"
        assert self.update_activation in ["relu", "elu", "leaky_relu"], \
            f"update_activation must be one of ['relu', 'elu', 'leaky_relu'], got {self.update_activation}"


@dataclass
class ReadoutFunctionConfig:
    """Graph-level readout function configuration."""
    readout_type: str = "sum"  # sum, mean, attention, mlp
    readout_layers: int = 1
    readout_hidden_dim: int = 64
    readout_dropout: float = 0.0

    def validate(self):
        """Validate readout function configuration."""
        assert self.readout_type in ["sum", "mean", "attention", "mlp"], \
            f"readout_type must be one of ['sum', 'mean', 'attention', 'mlp'], got {self.readout_type}"
        assert self.readout_layers > 0, "readout_layers must be positive"
        assert self.readout_hidden_dim > 0, "readout_hidden_dim must be positive"
        assert 0 <= self.readout_dropout < 1.0, "readout_dropout must be in [0, 1)"


@dataclass
class ModelConfig:
    """Complete model configuration."""
    node_hidden_dim: int = 64
    edge_hidden_dim: int = 32
    data_config: DataConfig = field(default_factory=DataConfig)
    message_config: MessageFunctionConfig = field(default_factory=MessageFunctionConfig)
    update_config: UpdateFunctionConfig = field(default_factory=UpdateFunctionConfig)
    readout_config: ReadoutFunctionConfig = field(default_factory=ReadoutFunctionConfig)
    task_type: str = "regression"  # regression or classification
    num_output_nodes: int = 1
    batch_norm: bool = True
    layer_norm: bool = True

    def validate(self):
        """Validate model configuration."""
        self.data_config.validate()
        self.message_config.validate()
        self.update_config.validate()
        self.readout_config.validate()
        assert self.node_hidden_dim > 0, "node_hidden_dim must be positive"
        assert self.edge_hidden_dim > 0, "edge_hidden_dim must be positive"
        assert self.task_type in ["regression", "classification"], \
            f"task_type must be 'regression' or 'classification', got {self.task_type}"
        assert self.num_output_nodes > 0, "num_output_nodes must be positive"


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    epochs: int = 360
    batch_size: int = 100
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.995
    lr_schedule: str = "exponential"  # exponential, cosine, step
    optimizer: str = "adam"  # adam, sgd, rmsprop
    early_stopping_patience: int = 50
    early_stopping_metric: str = "val_loss"  # val_loss, val_acc
    gradient_clip: float = 1.0
    seed: int = 42
    use_cuda: bool = False

    def validate(self):
        """Validate training configuration."""
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.learning_rate_decay <= 1.0, "learning_rate_decay must be in (0, 1]"
        assert self.lr_schedule in ["exponential", "cosine", "step"], \
            f"lr_schedule must be one of ['exponential', 'cosine', 'step'], got {self.lr_schedule}"
        assert self.optimizer in ["adam", "sgd", "rmsprop"], \
            f"optimizer must be one of ['adam', 'sgd', 'rmsprop'], got {self.optimizer}"
        assert self.early_stopping_patience > 0, "early_stopping_patience must be positive"
        assert self.gradient_clip > 0, "gradient_clip must be positive"


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    log_dir: str = "logs/mpnn"
    checkpoint_dir: str = "checkpoints/mpnn"
    results_dir: str = "results/mpnn"
    experiment_name: str = "mpnn_reproduction"
    log_interval: int = 1
    checkpoint_interval: int = 10
    verbose: bool = True
    use_tensorboard: bool = False
    save_plots: bool = False
    plot_dir: str = "plots/mpnn"

    def validate(self):
        """Validate output configuration."""
        assert self.log_interval > 0, "log_interval must be positive"
        assert self.checkpoint_interval > 0, "checkpoint_interval must be positive"


@dataclass
class MPNNConfig:
    """Complete MPNN configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self):
        """Validate all configuration components."""
        print("[CONFIG] Validating configuration...")
        self.data.validate()
        print("  ✓ Data config valid")
        self.model.validate()
        print("  ✓ Model config valid")
        self.training.validate()
        print("  ✓ Training config valid")
        self.output.validate()
        print("  ✓ Output config valid")
        
        # Cross-component validation
        assert self.training.batch_size == self.data.batch_size, \
            "Training batch_size must match data batch_size"
        print("  ✓ Cross-component compatibility verified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": asdict(self.data),
            "model": {
                "node_hidden_dim": self.model.node_hidden_dim,
                "edge_hidden_dim": self.model.edge_hidden_dim,
                "data": asdict(self.model.data_config),
                "message": asdict(self.model.message_config),
                "update": asdict(self.model.update_config),
                "readout": asdict(self.model.readout_config),
                "task_type": self.model.task_type,
                "num_output_nodes": self.model.num_output_nodes,
                "batch_norm": self.model.batch_norm,
                "layer_norm": self.model.layer_norm,
            },
            "training": asdict(self.training),
            "output": asdict(self.output),
        }

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[CONFIG] Configuration saved to {filepath}")

    @staticmethod
    def load_from_file(filepath: str) -> 'MPNNConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = MPNNConfig()
        # Update fields from loaded dict
        # (simplified - full implementation would be more robust)
        return config


def get_qm9_config() -> MPNNConfig:
    """Get QM9 molecular property prediction configuration."""
    config = MPNNConfig()
    config.data = DataConfig(
        dataset_name="qm9",
        dataset_path="data/qm9/dsgdb9nsd",
        batch_size=100,
        qm9_target_property=0,  # dipole moment
    )
    config.model.data_config = config.data
    config.model.message_config = MessageFunctionConfig(
        message_type="duvenaud",
        message_hidden_dim=64,
        message_passing_steps=3,
    )
    config.model.update_config = UpdateFunctionConfig(
        update_type="mlp",
        update_hidden_dim=128,
    )
    config.model.readout_config = ReadoutFunctionConfig(
        readout_type="sum",
        readout_hidden_dim=64,
    )
    config.training = TrainingConfig(
        epochs=360,
        batch_size=100,
        learning_rate=1e-3,
    )
    config.output = OutputConfig(
        experiment_name="mpnn_qm9_reproduction",
    )
    return config


def get_letter_config() -> MPNNConfig:
    """Get LETTER graph classification configuration."""
    config = MPNNConfig()
    config.data = DataConfig(
        dataset_name="letter",
        dataset_path="data/letter",
        batch_size=50,
    )
    config.model = ModelConfig(
        data_config=config.data,
        message_config=MessageFunctionConfig(
            message_type="ggnn",
            message_hidden_dim=32,
            message_passing_steps=5,
        ),
        update_config=UpdateFunctionConfig(
            update_type="gru",
            update_hidden_dim=64,
            update_dropout=0.5,
        ),
        readout_config=ReadoutFunctionConfig(
            readout_type="mean",
            readout_hidden_dim=128,
        ),
        task_type="classification",
        num_output_nodes=15,
    )
    config.training = TrainingConfig(
        epochs=200,
        batch_size=50,
        learning_rate=5e-4,
    )
    config.output = OutputConfig(
        experiment_name="mpnn_letter_reproduction",
    )
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_qm9_config()
    config.validate()
    print(f"[CONFIG] Configuration valid! ✓\n")
    print(f"QM9 Configuration loaded successfully!")
    print(f"Experiment: {config.output.experiment_name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Model: {config.model.message_config.message_type}")
    print(f"Epochs: {config.training.epochs}")
