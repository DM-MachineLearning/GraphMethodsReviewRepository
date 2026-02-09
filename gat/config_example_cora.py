"""
Example config for GAT on Cora.
"""

import os
import sys
from pathlib import Path

# Ensure package imports work when executed as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gat.config import GATConfig, DataConfig, ModelConfig, TrainingConfig, OutputConfig


def get_config() -> GATConfig:
    data = DataConfig(
        dataset_name="cora",
        dataset_path="data",
        batch_size=1,
        nhood=1,
    )

    model = ModelConfig(
        hid_units=[8],
        n_heads=[8, 1],
        residual=False,
        activation="elu",
        sparse=False,
    )

    training = TrainingConfig(
        epochs=100000,
        patience=100,
        learning_rate=0.005,
        l2_coef=0.0005,
        attn_drop=0.6,
        ffd_drop=0.6,
        seed=42,
    )

    output = OutputConfig(
        logs_dir="logs/gat",
        results_dir="results/gat",
        checkpoint_dir="checkpoints/gat",
        experiment_name="gat_cora",
        verbose=True,
    )

    return GATConfig(data=data, model=model, training=training, output=output)


if __name__ == "__main__":
    config = get_config()
    config.validate()
    print("✓ Cora config valid")
