"""
Helper script to run GAT with the default config.
Usage:
    python README_GAT_RUN.py
"""

from config_example_cora import get_config
from run_experiment import GATExperiment


def main():
    config = get_config()
    experiment = GATExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
