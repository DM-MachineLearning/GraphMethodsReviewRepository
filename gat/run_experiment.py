"""
GAT End-to-End Experiment Runner
"""

import os
import sys
from pathlib import Path
import json
import logging
import numpy as np
from datetime import datetime

import tensorflow as tf

# Enable TF 1.x behavior in TF 2.x
try:
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gat.models import GAT
from gat.utils import process
from gat.config import GATConfig, get_cora_config
from gat.data_loader import load_citation_data


class GATExperiment:
    """Complete GAT training and evaluation pipeline."""

    def __init__(self, config: GATConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}

    def _setup_logging(self) -> logging.Logger:
        os.makedirs(self.config.output.logs_dir, exist_ok=True)

        logger = logging.getLogger("GAT")
        logger.setLevel(logging.DEBUG)

        log_file = os.path.join(
            self.config.output.logs_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.config.output.verbose else logging.WARNING)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def run(self):
        try:
            self.logger.info("\n[1/6] Validating configuration...")
            self.config.validate()

            self.logger.info("\n[2/6] Loading data...")
            self.data = load_citation_data(self.config.data)

            self.logger.info("\n[3/6] Building model...")
            self._build_model()

            self.logger.info("\n[4/6] Training model...")
            self._train()

            self.logger.info("\n[5/6] Evaluating model...")
            self._evaluate()

            self.logger.info("\n[6/6] Saving results...")
            self._save_results()

            self.logger.info("\n✓ EXPERIMENT COMPLETED SUCCESSFULLY")
        except Exception as e:
            self.logger.error(f"✗ EXPERIMENT FAILED: {e}", exc_info=True)
            raise

    def _build_model(self):
        if hasattr(tf, "reset_default_graph"):
            tf.reset_default_graph()

        seed = self.config.training.seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        batch_size = self.config.data.batch_size
        nb_nodes = self.data["nb_nodes"]
        ft_size = self.data["ft_size"]
        nb_classes = self.data["nb_classes"]

        with tf.name_scope("input"):
            self.ftr_in = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, nb_nodes, ft_size))
            self.bias_in = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            self.lbl_in = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            self.msk_in = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, nb_nodes))
            self.attn_drop = tf.compat.v1.placeholder(tf.float32, shape=())
            self.ffd_drop = tf.compat.v1.placeholder(tf.float32, shape=())
            self.is_train = tf.compat.v1.placeholder(tf.bool, shape=())

        activation = tf.nn.elu if self.config.model.activation == "elu" else tf.nn.relu

        logits = GAT.inference(
            self.ftr_in,
            nb_classes,
            nb_nodes,
            self.is_train,
            self.attn_drop,
            self.ffd_drop,
            bias_mat=self.bias_in,
            hid_units=self.config.model.hid_units,
            n_heads=self.config.model.n_heads,
            residual=self.config.model.residual,
            activation=activation,
        )

        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(self.lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(self.msk_in, [-1])

        self.loss = GAT.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        self.accuracy = GAT.masked_accuracy(log_resh, lab_resh, msk_resh)
        self.train_op = GAT.training(self.loss, self.config.training.learning_rate, self.config.training.l2_coef)

        self.saver = tf.compat.v1.train.Saver()
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

    def _train(self):
        features = self.data["features"]
        biases = self.data["biases"]
        y_train = self.data["y_train"]
        y_val = self.data["y_val"]
        train_mask = self.data["train_mask"]
        val_mask = self.data["val_mask"]

        patience = self.config.training.patience
        epochs = self.config.training.epochs

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        with tf.compat.v1.Session() as sess:
            sess.run(self.init_op)

            for epoch in range(epochs):
                # Train
                _, loss_value_tr, acc_tr = sess.run(
                    [self.train_op, self.loss, self.accuracy],
                    feed_dict={
                        self.ftr_in: features,
                        self.bias_in: biases,
                        self.lbl_in: y_train,
                        self.msk_in: train_mask,
                        self.is_train: True,
                        self.attn_drop: self.config.training.attn_drop,
                        self.ffd_drop: self.config.training.ffd_drop,
                    },
                )

                # Val
                loss_value_vl, acc_vl = sess.run(
                    [self.loss, self.accuracy],
                    feed_dict={
                        self.ftr_in: features,
                        self.bias_in: biases,
                        self.lbl_in: y_val,
                        self.msk_in: val_mask,
                        self.is_train: False,
                        self.attn_drop: 0.0,
                        self.ffd_drop: 0.0,
                    },
                )

                if epoch % 10 == 0:
                    self.logger.info(
                        "Epoch %d | train_loss: %.5f acc: %.5f | val_loss: %.5f acc: %.5f",
                        epoch,
                        loss_value_tr,
                        acc_tr,
                        loss_value_vl,
                        acc_vl,
                    )

                if acc_vl >= vacc_mx or loss_value_vl <= vlss_mn:
                    if acc_vl >= vacc_mx and loss_value_vl <= vlss_mn:
                        vacc_early_model = acc_vl
                        vlss_early_model = loss_value_vl
                        ckpt_dir = self.config.output.checkpoint_dir
                        os.makedirs(ckpt_dir, exist_ok=True)
                        ckpt_file = os.path.join(ckpt_dir, f"{self.config.output.experiment_name}.ckpt")
                        self.saver.save(sess, ckpt_file)
                    vacc_mx = np.max((acc_vl, vacc_mx))
                    vlss_mn = np.min((loss_value_vl, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        self.logger.info("Early stop! Min loss: %.5f, Max accuracy: %.5f", vlss_mn, vacc_mx)
                        self.logger.info("Early stop model val loss: %.5f, acc: %.5f", vlss_early_model, vacc_early_model)
                        break

            self.session = sess

    def _evaluate(self):
        features = self.data["features"]
        biases = self.data["biases"]
        y_test = self.data["y_test"]
        test_mask = self.data["test_mask"]

        loss_value_ts, acc_ts = self.session.run(
            [self.loss, self.accuracy],
            feed_dict={
                self.ftr_in: features,
                self.bias_in: biases,
                self.lbl_in: y_test,
                self.msk_in: test_mask,
                self.is_train: False,
                self.attn_drop: 0.0,
                self.ffd_drop: 0.0,
            },
        )

        self.results["test_loss"] = float(loss_value_ts)
        self.results["test_accuracy"] = float(acc_ts)
        self.logger.info("Test loss: %.5f | Test accuracy: %.5f", loss_value_ts, acc_ts)

    def _save_results(self):
        os.makedirs(self.config.output.results_dir, exist_ok=True)
        results_file = os.path.join(
            self.config.output.results_dir,
            f"{self.config.output.experiment_name}_results.json",
        )

        results = {
            "config": self.config.to_dict(),
            "results": self.results,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")


def main():
    config = get_cora_config()
    experiment = GATExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
