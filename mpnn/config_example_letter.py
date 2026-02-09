"""
LETTER Configuration Example for MPNN
Graph classification on LETTER dataset.
"""

from config import (
    MPNNConfig, DataConfig, MessageFunctionConfig,
    UpdateFunctionConfig, ReadoutFunctionConfig, ModelConfig,
    TrainingConfig, OutputConfig
)


def get_config() -> MPNNConfig:
    """Get LETTER configuration for MPNN."""
    
    # Data configuration
    data_config = DataConfig(
        dataset_name="letter",
        dataset_path="data/letter",
        batch_size=50,
        num_workers=0,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
        shuffle=True,
        qm9_target_property=0,  # Not used for LETTER
        edge_representation="features",
        normalize_data=True,
    )

    # Message function configuration
    message_config = MessageFunctionConfig(
        message_type="ggnn",  # Try: duvenaud, ggnn, intnet
        message_hidden_dim=32,
        message_passing_steps=5,
    )

    # Update function configuration
    update_config = UpdateFunctionConfig(
        update_type="gru",  # Try: mlp, gru, lstm
        update_hidden_dim=64,
        update_dropout=0.5,
        update_activation="relu",
    )

    # Readout function configuration
    readout_config = ReadoutFunctionConfig(
        readout_type="mean",  # Try: sum, mean, attention, mlp
        readout_layers=2,
        readout_hidden_dim=128,
        readout_dropout=0.2,
    )

    # Model configuration
    model_config = ModelConfig(
        node_hidden_dim=64,
        edge_hidden_dim=32,
        data_config=data_config,
        message_config=message_config,
        update_config=update_config,
        readout_config=readout_config,
        task_type="classification",
        num_output_nodes=15,  # 15 letter classes
        batch_norm=True,
        layer_norm=True,
    )

    # Training configuration
    training_config = TrainingConfig(
        epochs=200,
        batch_size=50,
        learning_rate=5e-4,
        learning_rate_decay=0.99,
        lr_schedule="exponential",
        optimizer="adam",
        early_stopping_patience=30,
        early_stopping_metric="val_loss",
        gradient_clip=1.0,
        seed=42,
        use_cuda=False,
    )

    # Output configuration
    output_config = OutputConfig(
        log_dir="logs/mpnn",
        checkpoint_dir="checkpoints/mpnn",
        results_dir="results/mpnn",
        experiment_name="mpnn_letter_reproduction",
        log_interval=1,
        checkpoint_interval=10,
        verbose=True,
        use_tensorboard=False,
        save_plots=False,
    )

    # Complete configuration
    config = MPNNConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        output=output_config,
    )

    return config


if __name__ == "__main__":
    config = get_config()
    config.validate()
    print("✓ LETTER configuration loaded successfully")
