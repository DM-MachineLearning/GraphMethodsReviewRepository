"""
GCN Configuration Template - Citeseer Dataset
==============================================
Example configuration for Citeseer citation network.

Citeseer Dataset Info:
- Nodes: 3,327 papers
- Edges: 4,732 citations
- Features: 3,703 words (binary bag-of-words)
- Classes: 6 computer science topics
- Standard splits: Similar to Cora but scaled for size
"""

from config import (
    GCNConfig, DataConfig, GraphConfig, ModelConfig, 
    TrainingConfig, OutputConfig, RegularizationConfig
)


# Dataset configuration
data_config = DataConfig(
    dataset_name='citeseer',
    dataset_path='./gcn/data',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    normalize_features=True,
    sparse_features=True,
)

# Graph configuration
graph_config = GraphConfig(
    add_self_loops=True,
    normalize_adj=True,
    symmetric_normalize=True,
    max_degree=3,
)

# Model configuration (similar to Cora)
model_config = ModelConfig(
    model_type='gcn',
    hidden1=16,  # Same as Cora
    hidden2=None,
    dropout=0.5,
    weight_decay=5e-4,
    activation='relu',
)

# Training configuration
training_config = TrainingConfig(
    epochs=200,
    learning_rate=0.01,
    early_stopping=10,
    early_stopping_metric='val_loss',
    batch_size=None,
    optimizer='adam',
    lr_decay=False,
    lr_decay_rate=0.95,
    lr_decay_steps=10,
    seed=123,
)

# Output configuration
output_config = OutputConfig(
    output_dir='./outputs',
    checkpoints_dir='./outputs/checkpoints',
    logs_dir='./outputs/logs',
    results_dir='./outputs/results',
    save_model=True,
    save_results=True,
    save_config=True,
    verbose=True,
    log_every_n_steps=10,
)

# Regularization configuration
regularization_config = RegularizationConfig(
    spatial_dropout=False,
    batch_norm=False,
    label_smoothing=0.0,
)

# Complete configuration
config = GCNConfig(
    data=data_config,
    graph=graph_config,
    model=model_config,
    training=training_config,
    output=output_config,
    regularization=regularization_config,
)


if __name__ == '__main__':
    print("=" * 70)
    print("GCN Configuration - Citeseer Dataset")
    print("=" * 70)
    print("\nDataset: Citeseer")
    print("  - 3,327 nodes (papers)")
    print("  - 4,732 edges (citations)")
    print("  - 3,703 features (bag-of-words)")
    print("  - 6 classes (CS topics)")
    print("\nModel: GCN (same settings as Cora)")
    print(f"  - Hidden units: {model_config.hidden1}")
    print(f"  - Dropout: {model_config.dropout}")
    print(f"  - Weight decay: {model_config.weight_decay}")
    print("=" * 70)
    
    config.validate()
    print("\nConfiguration valid ✓\n")
