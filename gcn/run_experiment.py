"""
GCN End-to-End Experiment Runner
================================
Complete pipeline for training and evaluating Graph Convolutional Networks.

Paper: Kipf & Welling (2017) - Semi-Supervised Classification with GCN
       https://arxiv.org/abs/1609.02907

This script orchestrates:
1. Configuration loading and validation
2. Data loading and preprocessing
3. Model building and initialization
4. Training with validation and early stopping
5. Evaluation on test set
6. Results saving and logging
"""

import os
import sys
import time
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import get_cora_config, GCNConfig
from utils import load_data, preprocess_features, preprocess_adj, chebyshev_polynomials
from models import GCN, MLP


class GCNExperiment:
    """Complete GCN training and evaluation pipeline."""
    
    def __init__(self, config: GCNConfig):
        """Initialize experiment with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.session = None
        self.model = None
        self.train_op = None
        self.results = {}
        
        self.logger.info("=" * 70)
        self.logger.info("GCN EXPERIMENT INITIALIZATION")
        self.logger.info("=" * 70)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        os.makedirs(self.config.output.logs_dir, exist_ok=True)
        
        logger = logging.getLogger('GCN')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = os.path.join(
            self.config.output.logs_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.config.output.verbose else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def run(self):
        """Execute complete pipeline."""
        try:
            # 1. Configuration validation
            self.logger.info("\n[1/6] Validating configuration...")
            self.config.validate()
            self._log_config()
            
            # 2. Data loading
            self.logger.info("\n[2/6] Loading and preprocessing data...")
            self._load_data()
            
            # 3. Build model
            self.logger.info("\n[3/6] Building model...")
            self._build_model()
            
            # 4. Training
            self.logger.info("\n[4/6] Training model...")
            self._train()
            
            # 5. Evaluation
            self.logger.info("\n[5/6] Evaluating model...")
            self._evaluate()
            
            # 6. Save results
            self.logger.info("\n[6/6] Saving results...")
            self._save_results()
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"\n✗ EXPERIMENT FAILED: {str(e)}", exc_info=True)
            raise
        finally:
            if self.session:
                self.session.close()
    
    def _log_config(self):
        """Log configuration details."""
        config_dict = self.config.to_dict()
        self.logger.info(f"\nConfiguration:")
        for section, values in config_dict.items():
            self.logger.info(f"  {section.upper()}:")
            for key, value in values.items():
                self.logger.info(f"    {key}: {value}")
    
    def _load_data(self):
        """Load and preprocess data."""
        self.logger.info(f"Loading dataset: {self.config.data.dataset_name}")
        
        # Load from Kipf implementation
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
            self.config.data.dataset_name
        )
        
        self.logger.info(f"  Adjacency matrix shape: {adj.shape}")
        self.logger.info(f"  Features shape: {features.shape}")
        self.logger.info(f"  Train/val/test splits: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
        self.logger.info(f"  Number of classes: {y_train.shape[1]}")
        
        # Preprocess features
        features = preprocess_features(features)
        self.logger.info(f"  Features preprocessed, type: {type(features)}")
        
        # Preprocess adjacency matrix
        if self.config.model.model_type == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            self.logger.info(f"  Using standard GCN (1 support matrix)")
        elif self.config.model.model_type == 'gcn_cheby':
            support = chebyshev_polynomials(adj, self.config.graph.max_degree)
            num_supports = 1 + self.config.graph.max_degree
            self.logger.info(f"  Using Chebyshev GCN with degree {self.config.graph.max_degree}")
        else:  # dense
            support = [preprocess_adj(adj)]
            num_supports = 1
            self.logger.info(f"  Using dense model")
        
        # Store data
        self.adj = adj
        self.features = features
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.support = support
        self.num_supports = num_supports
        self.num_nodes = adj.shape[0]
        self.num_features = features.shape[1]
        self.num_classes = y_train.shape[1]
        
        self.logger.info(f"✓ Data loaded successfully")
    
    def _build_model(self):
        """Build TensorFlow model."""
        tf.reset_default_graph()
        
        # Set random seed
        seed = self.config.training.seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(self.num_supports)],
            'features': tf.sparse_placeholder(tf.float32),
            'labels': tf.placeholder(tf.float32, shape=(None, self.num_classes)),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32),
        }
        
        self.logger.info(f"  Model type: {self.config.model.model_type}")
        self.logger.info(f"  Hidden units: {self.config.model.hidden1}")
        self.logger.info(f"  Dropout rate: {self.config.model.dropout}")
        self.logger.info(f"  Weight decay: {self.config.model.weight_decay}")
        
        # Build model
        if self.config.model.model_type in ['gcn', 'gcn_cheby']:
            model = GCN(
                placeholders,
                self.num_features,
                self.num_classes,
                self.num_supports,
                self.config.model.hidden1,
                act=tf.nn.relu,
                dropout=True,
                sparse_inputs=True,
                featureless=False,
                logging=self.config.output.verbose,
            )
        else:  # dense
            model = MLP(
                placeholders,
                self.num_features,
                self.num_classes,
                self.num_supports,
                self.config.model.hidden1,
                act=tf.nn.relu,
                dropout=True,
                sparse_inputs=False,
                featureless=False,
                logging=self.config.output.verbose,
            )
        
        self.model = model
        self.placeholders = placeholders
        
        # Build optimization
        with tf.name_scope('optimizer'):
            # Masked loss
            model.loss = tf.losses.softmax_cross_entropy(
                self.placeholders['labels'],
                model.outputs,
                weights=tf.cast(self.placeholders['labels_mask'], tf.float32)
            )
            
            # L2 regularization
            for var in tf.trainable_variables():
                model.loss += self.config.model.weight_decay * tf.nn.l2_loss(var)
            
            # Optimizer
            if self.config.training.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.config.training.learning_rate
                )
            else:  # sgd
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.config.training.learning_rate
                )
            
            self.train_op = optimizer.minimize(model.loss)
        
        # Accuracy
        model.accuracy = tf.contrib.metrics.accuracy(
            tf.argmax(self.placeholders['labels'], axis=1),
            tf.argmax(model.outputs, axis=1),
            weights=tf.cast(self.placeholders['labels_mask'], tf.float32)
        )
        
        self.logger.info(f"✓ Model built successfully")
    
    def _train(self):
        """Train model with validation and early stopping."""
        # Start session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
        # Prepare feed dict function
        def construct_feed_dict(features, support, labels, labels_mask, dropout=0.):
            """Construct feed dictionary for model."""
            feed_dict = dict()
            feed_dict.update({self.placeholders['labels']: labels})
            feed_dict.update({self.placeholders['labels_mask']: labels_mask})
            feed_dict.update({self.placeholders['dropout']: dropout})
            feed_dict.update({self.placeholders['num_features_nonzero']: features[1].shape})
            
            for i in range(len(support)):
                feed_dict.update({self.placeholders['support'][i]: support[i]})
            
            feed_dict.update({self.placeholders['features']: features})
            return feed_dict
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        val_accs = []
        
        self.logger.info(f"\nTraining for {self.config.training.epochs} epochs...")
        self.logger.info(f"Early stopping patience: {self.config.training.early_stopping}")
        
        for epoch in range(self.config.training.epochs):
            t = time.time()
            
            # Training step
            feed_dict_train = construct_feed_dict(
                self.features,
                self.support,
                self.y_train,
                self.train_mask,
                self.config.model.dropout
            )
            
            _, train_loss, train_acc = self.session.run(
                [self.train_op, self.model.loss, self.model.accuracy],
                feed_dict=feed_dict_train
            )
            
            # Validation step
            feed_dict_val = construct_feed_dict(
                self.features,
                self.support,
                self.y_val,
                self.val_mask,
                0.0
            )
            
            val_loss, val_acc = self.session.run(
                [self.model.loss, self.model.accuracy],
                feed_dict=feed_dict_val
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Logging
            if epoch % self.config.output.log_every_n_steps == 0 or epoch == self.config.training.epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
                    f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                    f"val_acc: {val_acc:.4f} | time: {time.time()-t:.2f}s"
                )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            
            if epoch - best_epoch >= self.config.training.early_stopping:
                self.logger.info(
                    f"\nEarly stopping at epoch {epoch+1} (best validation loss: {best_val_loss:.4f})"
                )
                break
        
        # Store training history
        self.results['train_losses'] = train_losses
        self.results['val_losses'] = val_losses
        self.results['val_accs'] = val_accs
        self.results['best_epoch'] = best_epoch + 1
        self.results['best_val_loss'] = float(best_val_loss)
        self.results['best_val_acc'] = float(max(val_accs))
        
        self.logger.info(f"✓ Training completed (best epoch: {best_epoch + 1})")
    
    def _evaluate(self):
        """Evaluate on test set."""
        feed_dict_test = {
            self.placeholders['support']: self.support,
            self.placeholders['features']: self.features,
            self.placeholders['labels']: self.y_test,
            self.placeholders['labels_mask']: self.test_mask,
            self.placeholders['dropout']: 0.,
            self.placeholders['num_features_nonzero']: self.features[1].shape,
        }
        
        test_loss, test_acc = self.session.run(
            [self.model.loss, self.model.accuracy],
            feed_dict=feed_dict_test
        )
        
        self.logger.info(f"\nTest Results:")
        self.logger.info(f"  Test loss: {test_loss:.4f}")
        self.logger.info(f"  Test accuracy: {test_acc:.4f}")
        
        # Store results
        self.results['test_loss'] = float(test_loss)
        self.results['test_accuracy'] = float(test_acc)
        
        self.logger.info(f"✓ Evaluation completed")
    
    def _save_results(self):
        """Save results to files."""
        os.makedirs(self.config.output.results_dir, exist_ok=True)
        
        # Save results JSON
        results_file = os.path.join(
            self.config.output.results_dir,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.config.data.dataset_name,
            'model': self.config.model.model_type,
            'configuration': self.config.to_dict(),
            'results': self.results,
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        self.logger.info(f"\n✓ Results saved to: {results_file}")
        self.logger.info(f"  File size: {os.path.getsize(results_file) / 1024:.1f} KB")
        
        # Save config JSON
        if self.config.output.save_config:
            config_file = os.path.join(
                self.config.output.results_dir,
                f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.config.to_json(config_file)
            self.logger.info(f"✓ Configuration saved to: {config_file}")


def main():
    """Main entry point."""
    # Get default Cora configuration
    config = get_cora_config()
    
    # Run experiment
    experiment = GCNExperiment(config)
    experiment.run()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Model: {config.model.model_type}")
    print(f"Results: {experiment.results}")
    print("=" * 70)


if __name__ == '__main__':
    main()
