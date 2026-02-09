"""
GAT Configuration System
Configuration dataclasses for Graph Attention Networks experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "cora"  # cora, citeseer, pubmed
    dataset_path: str = "data"
    batch_size: int = 1
    nhood: int = 1  # neighborhood size for bias

    def validate(self):
        assert self.dataset_name in ["cora", "citeseer", "pubmed"], (
            f"dataset_name must be one of cora/citeseer/pubmed, got {self.dataset_name}"
        )
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.nhood >= 1, "nhood must be >= 1"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hid_units: List[int] = field(default_factory=lambda: [8])
    n_heads: List[int] = field(default_factory=lambda: [8, 1])
    residual: bool = False
    activation: str = "elu"  # elu, relu
    sparse: bool = False

    def validate(self):
        assert len(self.n_heads) == len(self.hid_units) + 1, (
            "n_heads must have len(hid_units)+1 (output heads)"
        )
        assert all(h > 0 for h in self.hid_units), "hid_units must be positive"
        assert all(h > 0 for h in self.n_heads), "n_heads must be positive"
        assert self.activation in ["elu", "relu"], "activation must be elu or relu"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100000
    patience: int = 100
    learning_rate: float = 0.005
    l2_coef: float = 0.0005
    attn_drop: float = 0.6
    ffd_drop: float = 0.6
    seed: int = 42

    def validate(self):
        assert self.epochs > 0, "epochs must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.l2_coef >= 0, "l2_coef must be non-negative"
        assert 0 <= self.attn_drop < 1, "attn_drop must be in [0,1)"
        assert 0 <= self.ffd_drop < 1, "ffd_drop must be in [0,1)"


@dataclass
class OutputConfig:
    """Output and checkpoint configuration."""
    logs_dir: str = "logs/gat"
    results_dir: str = "results/gat"
    checkpoint_dir: str = "checkpoints/gat"
    experiment_name: str = "gat_cora"
    verbose: bool = True

    def validate(self):
        assert self.experiment_name, "experiment_name must be set"


@dataclass
class GATConfig:
    """Full configuration wrapper."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self):
        print("[CONFIG] Validating configuration...")
        self.data.validate()
        print("  ✓ Data config valid")
        self.model.validate()
        print("  ✓ Model config valid")
        self.training.validate()
        print("  ✓ Training config valid")
        self.output.validate()
        print("  ✓ Output config valid")

        if self.model.sparse:
            assert self.data.batch_size == 1, "Sparse GAT requires batch_size=1"
        print("  ✓ Cross-component compatibility verified")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "output": asdict(self.output),
        }

    def save_to_file(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_cora_config() -> GATConfig:
    """Default Cora configuration."""
    config = GATConfig()
    config.data = DataConfig(dataset_name="cora", dataset_path="data", batch_size=1, nhood=1)
    config.model = ModelConfig(hid_units=[8], n_heads=[8, 1], residual=False, activation="elu", sparse=False)
    config.training = TrainingConfig(epochs=100000, patience=100, learning_rate=0.005, l2_coef=0.0005)
    config.output = OutputConfig(experiment_name="gat_cora")
    return config


if __name__ == "__main__":
    config = get_cora_config()
    config.validate()
    print("[CONFIG] Configuration valid! ✓")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Heads: {config.model.n_heads}")
    print(f"Hidden units: {config.model.hid_units}")
