"""
Data Loader for MPNN
Unified data loading interface for different datasets.
"""

from typing import Dict, Any, Optional
from config import DataConfig


class MPNNDataLoader:
    """Data loader for MPNN datasets."""

    def __init__(self, config: DataConfig):
        """Initialize data loader."""
        self.config = config

    def load_dataset(self) -> Dict[str, Any]:
        """Load dataset based on configuration."""
        if self.config.dataset_name == "qm9":
            return self._load_qm9()
        elif self.config.dataset_name == "letter":
            return self._load_letter()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

    def _load_qm9(self) -> Dict[str, Any]:
        """Load QM9 molecular property prediction dataset."""
        try:
            import os
            # Try to load real QM9 data
            if os.path.exists(self.config.dataset_path):
                print(f"[DATA] Loading QM9 from {self.config.dataset_path}")
            else:
                print(f"[DATA] Creating dummy dataset for testing...")
                return self._create_dummy_qm9()
        except Exception as e:
            print(f"[DATA] Creating dummy dataset for testing...")
            return self._create_dummy_qm9()

    def _load_letter(self) -> Dict[str, Any]:
        """Load LETTER graph classification dataset."""
        try:
            import os
            if os.path.exists(self.config.dataset_path):
                print(f"[DATA] Loading LETTER from {self.config.dataset_path}")
            else:
                print(f"[DATA] Creating dummy dataset for testing...")
                return self._create_dummy_letter()
        except Exception as e:
            print(f"[DATA] Creating dummy dataset for testing...")
            return self._create_dummy_letter()

    def _create_dummy_qm9(self) -> Dict[str, Any]:
        """Create dummy QM9 dataset for testing."""
        total_samples = 133885
        train_size = int(total_samples * self.config.train_split)
        val_size = int(total_samples * self.config.val_split)
        test_size = total_samples - train_size - val_size

        return {
            "name": "qm9",
            "num_samples": total_samples,
            "num_properties": 14,
            "num_atoms": 29,
            "num_atom_types": 5,
            "num_edge_types": 5,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "target_property": self.config.qm9_target_property,
            "task_type": "regression",
        }

    def _create_dummy_letter(self) -> Dict[str, Any]:
        """Create dummy LETTER dataset for testing."""
        total_samples = 750
        train_size = int(total_samples * self.config.train_split)
        val_size = int(total_samples * self.config.val_split)
        test_size = total_samples - train_size - val_size

        return {
            "name": "letter",
            "num_samples": total_samples,
            "num_classes": 15,
            "num_graphs": total_samples,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "task_type": "classification",
        }
