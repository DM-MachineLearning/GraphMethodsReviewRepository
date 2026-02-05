#!/usr/bin/env python3
"""
Setup and initialization script.

Performs initial setup:
1. Check dependencies
2. Create output directories
3. Validate configuration
4. Download/prepare example data (optional)

Usage:
    python setup.py

Options:
    python setup.py --skip-data     Skip data preparation
    python setup.py --validate-only Just validate config
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.6+"""
    if sys.version_info < (3, 6):
        print("✗ Python 3.6+ required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")


def check_dependencies():
    """Run dependency check."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    try:
        # Import and run check directly instead of subprocess
        from check_dependencies import check_package, print_status, REQUIRED_PACKAGES, OPTIONAL_PACKAGES
        
        results = []
        for package, min_version in REQUIRED_PACKAGES.items():
            results.append(check_package(package, min_version, required=True))
        for package, min_version in OPTIONAL_PACKAGES.items():
            results.append(check_package(package, min_version, required=False))
        
        return print_status(results)
    
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_directories():
    """Create necessary directories."""
    print("\n" + "="*80)
    print("CREATING DIRECTORIES")
    print("="*80)
    
    dirs = {
        'outputs': './outputs',
        'outputs/checkpoints': './outputs/checkpoints',
        'outputs/summaries': './outputs/summaries',
        'outputs/logs': './outputs/logs',
        'outputs/results': './outputs/results',
        'data': './data',
    }
    
    for name, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {path}")


def validate_config():
    """Validate configuration."""
    print("\n" + "="*80)
    print("VALIDATING CONFIGURATION")
    print("="*80)
    
    try:
        from config import validate_config as validate
        validate()
        print("✓ Configuration valid")
        return True
    
    except ImportError:
        print("✗ config.py not found")
        print("  Copy an example: cp config_example_custom.py config.py")
        return False
    
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def prepare_example_data():
    """Optionally prepare MNIST example data."""
    print("\n" + "="*80)
    print("EXAMPLE DATA")
    print("="*80)
    
    print("\nNo example data included. To test the system:")
    print("\n1. Use your own data:")
    print("   - Save features to: ./data/features.npz")
    print("   - Save labels to: ./data/labels.npy")
    print("   - Update config.py with paths")
    print("   - Run: python run_experiment.py")
    print("\n2. Or download public datasets:")
    print("   - MNIST: Use keras/tensorflow utilities")
    print("   - 20NEWS: Use scikit-learn datasets.fetch_20newsgroups()")


def create_example_notebooks():
    """Create example notebooks (optional)."""
    print("\n" + "="*80)
    print("EXAMPLE NOTEBOOKS")
    print("="*80)
    print("\nNotebooks available in repository:")
    print("  - rcv1.ipynb: RCV1 text classification example")
    print("  - usage.ipynb: Usage examples and tutorials")
    print("  - nips2016/mnist.ipynb: MNIST classification")
    print("  - nips2016/20news.ipynb: 20 newsgroups classification")
    print("  - trials/*: Additional experiments")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("""
1. UNDERSTAND THE SYSTEM:
   - Read the README: cat README_REPRODUCIBLE.md
   - Review config: cat config.py

2. PREPARE YOUR DATA:
   - Place features in: ./data/features.npz (shape: N×F)
   - Place labels in: ./data/labels.npy (shape: N,)
   - Or use existing notebooks for example data

3. CONFIGURE:
   - Copy example: cp config_example_custom.py config.py
   - Edit settings: nano config.py
   - For MNIST: cp config_example_mnist.py config.py
   - For text: cp config_example_20news.py config.py

4. RUN EXPERIMENT:
   - Execute: python run_experiment.py
   - Monitor: Check ./outputs/logs/training.log
   - View results: cat ./outputs/results/results.json

5. TUNE HYPERPARAMETERS:
   - Edit config.py parameters
   - Run again with new settings
   - Compare results in ./outputs/

RESOURCES:
  - Complete guide: README_REPRODUCIBLE.md
  - Setup guide: SETUP_GUIDE.md
  - Implementation details: IMPLEMENTATION_SUMMARY.md
  - Quick reference: QUICK_REFERENCE.md
""")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup CNN Graph environment')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    parser.add_argument('--validate-only', action='store_true', help='Only validate config')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CNN ON GRAPHS - SETUP SCRIPT")
    print("="*80 + "\n")
    
    # Check Python
    check_python_version()
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Setup failed: install missing dependencies")
        sys.exit(1)
    
    # If only validating, stop here
    if args.validate_only:
        if validate_config():
            print("\n✓ Setup validation successful")
            sys.exit(0)
        else:
            print("\n✗ Setup validation failed")
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Validate configuration
    if not validate_config():
        print("\n⚠ Configuration not valid. Create config.py:")
        print("  cp config_example_custom.py config.py")
        print("  nano config.py")
    
    # Prepare data
    if not args.skip_data:
        prepare_example_data()
    
    # Notebooks
    create_example_notebooks()
    
    # Next steps
    print_next_steps()
    
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
