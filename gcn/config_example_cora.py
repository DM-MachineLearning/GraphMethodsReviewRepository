"""
GCN Configuration Template - Cora Dataset
=========================================
Example configuration for Cora citation network.

Cora Dataset Info:
- Nodes: 2,708 papers
- Edges: 5,429 citations
- Features: 1,433 words (binary bag-of-words)
- Classes: 7 machine learning topics
- Standard splits: 140 train, 500 val, 1000+ test
"""

from config import (
    GCNConfig, DataConfig, GraphConfig, ModelConfig, 
    TrainingConfig, OutputConfig, RegularizationConfig
)


# Dataset configuration
data_config = DataConfig(
    dataset_name='cora',
    dataset_path='./data',
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

# Model configuration (from paper)
model_config = ModelConfig(
    model_type='gcn',  # Use standard GCN (not Chebyshev)
    hidden1=16,  # Hidden layer size from Kipf & Welling 2017
    hidden2=None,
    dropout=0.5,  # Dropout rate from paper
    weight_decay=5e-4,  # L2 regularization from paper
    activation='relu',
)

# Training configuration (from paper)
training_config = TrainingConfig(
    epochs=200,  # From paper
    learning_rate=0.01,  # From paper
    early_stopping=10,
    early_stopping_metric='val_loss',
    batch_size=None,  # Full batch as in paper
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
    print("GCN Configuration - Cora Dataset (Paper Settings)")
    print("=" * 70)
    print("\nDataset: Cora")
    print("  - 2,708 nodes (papers)")
    print("  - 5,429 edges (citations)")
    print("  - 1,433 features (bag-of-words)")
    print("  - 7 classes (ML topics)")
    print("\nModel: GCN")
    print(f"  - Hidden units: {model_config.hidden1}")
    print(f"  - Dropout: {model_config.dropout}")
    print(f"  - Weight decay: {model_config.weight_decay}")
    print("\nTraining:")
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - Early stopping: {training_config.early_stopping}")
    print("=" * 70)
    
    # Validate and print
    config.validate()
    print("\nConfiguration valid ✓\n")
