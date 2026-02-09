# MPNN Quick Start Guide

**Status:** ✅ Ready to use  
**Location:** `/home/dmlab/GraphMethodsReviewRepository/mpnn/`

---

## Running MPNN in 30 Seconds

### Option 1: Test with QM9 (Molecular Property Prediction)
```bash
cd /home/dmlab/GraphMethodsReviewRepository
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py
```

### Option 2: Test with LETTER (Graph Classification)
```bash
cd /home/dmlab/GraphMethodsReviewRepository
python mpnn/run_experiment.py --config mpnn/config_example_letter.py
```

### Option 3: Custom Parameters
```bash
cd /home/dmlab/GraphMethodsReviewRepository
python mpnn/run_experiment.py \
  --dataset qm9 \
  --message-type ggnn \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4
```

---

## What Gets Output

After running, you'll get:

✅ **Log File**
```
logs/mpnn/qm9/mpnn_qm9_reproduction_YYYYMMDD_HHMMSS.log
```

✅ **Results JSON** (with all metrics)
```
results/mpnn/qm9/mpnn_qm9_reproduction_results.json
```

✅ **Screen Output** (6 steps)
```
[1/6] Validating configuration... ✓
[2/6] Loading and preprocessing data... ✓
[3/6] Building model... ✓
[4/6] Training... ✓
[5/6] Evaluating on test set... ✓
[6/6] Saving results... ✓

✓ EXPERIMENT COMPLETED SUCCESSFULLY
```

---

## File Guide

| File | Purpose | Lines |
|------|---------|-------|
| `config.py` | Configuration system | 537 |
| `config_example_qm9.py` | QM9 config example | 250 |
| `config_example_letter.py` | LETTER config example | 230 |
| `data_loader.py` | Dataset handling | 150 |
| `run_experiment.py` | Training pipeline | 390 |
| `README_REPRODUCIBLE.md` | Complete guide | 450+ |
| `QUICK_REFERENCE.md` | Parameter reference | 150+ |

---

## Configuration Options

### Message Types
- `duvenaud` - Simple message passing
- `ggnn` - Gated Graph Neural Networks
- `intnet` - Interaction Networks

### Update Functions
- `mlp` - Multi-layer perceptron (fast)
- `gru` - Gated recurrent unit
- `lstm` - Long short-term memory (slower)

### Readout Types
- `sum` - Sum aggregation
- `mean` - Mean aggregation
- `attention` - Attention-based
- `mlp` - MLP aggregation

### Datasets
- `qm9` - Molecular properties (133k molecules)
- `letter` - Graph classification (750 graphs)

---

## Example: Custom QM9 Run

```bash
python mpnn/run_experiment.py \
  --config mpnn/config_example_qm9.py \
  --message-type ggnn \
  --epochs 200 \
  --batch-size 50 \
  --learning-rate 5e-4
```

---

## Understanding Results

### Results JSON File Contains:
- Complete configuration used
- Dataset information
- Training metrics (losses, accuracies per epoch)
- Best epoch information
- Test set metrics
- Timing information

### Key Metrics
- `train_loss` - Training loss
- `val_loss` - Validation loss
- `val_acc` - Validation accuracy
- `test_loss` - Test set loss
- `test_accuracy` - Test set accuracy
- `mae` - Mean absolute error
- `rmse` - Root mean square error

---

## Documentation Map

**Quick Questions?**
1. **"How do I run it?"** → This file ✓
2. **"What parameters exist?"** → `QUICK_REFERENCE.md`
3. **"I have a problem"** → `README_REPRODUCIBLE.md` (Troubleshooting section)
4. **"Show me an example"** → `config_example_*.py`

---

## All Available Commands

```bash
# Basic QM9 training
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py

# Basic LETTER training
python mpnn/run_experiment.py --config mpnn/config_example_letter.py

# Override dataset
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py --dataset qm9

# Override message type
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py --message-type ggnn

# Override epochs
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py --epochs 150

# Override batch size
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py --batch-size 128

# Override learning rate
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py --learning-rate 1e-3

# Combine multiple overrides
python mpnn/run_experiment.py --config mpnn/config_example_qm9.py \
  --message-type intnet \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 5e-5
```

---

## Expected Results

### QM9 Dipole Moment Prediction
- Typical MAE: ~0.05 Debye (with real data)
- Convergence: 100-150 epochs
- Test Accuracy: ~90-95%

### LETTER Graph Classification
- Duvenaud: ~93% accuracy
- GGNN: ~95% accuracy
- Convergence: 50-100 epochs

---

## Troubleshooting

### Q: Command not found
**A:** Make sure you're in the right directory:
```bash
cd /home/dmlab/GraphMethodsReviewRepository
```

### Q: Module not found
**A:** Dependencies need to be installed:
```bash
pip install torch numpy networkx joblib
```

### Q: Permission denied
**A:** Check file permissions:
```bash
chmod +x mpnn/run_experiment.py
```

### Q: Results file not created
**A:** Check the output directory exists:
```bash
mkdir -p results/mpnn/qm9
```

---

## Next Steps

1. **Run a quick test:** `python mpnn/run_experiment.py --config mpnn/config_example_qm9.py`
2. **Check results:** `cat results/mpnn/qm9/mpnn_qm9_*_results.json | python -m json.tool`
3. **Read full docs:** See `mpnn/README_REPRODUCIBLE.md`
4. **Try custom config:** Modify `config_example_*.py` and test

---

## Summary

✅ MPNN is ready to use with working examples  
✅ Configuration system validated  
✅ Training pipeline tested end-to-end  
✅ Results saved in JSON format  
✅ Full documentation available  

**Start with:** `python mpnn/run_experiment.py --config mpnn/config_example_qm9.py`

---

For detailed information, see:
- [README_REPRODUCIBLE.md](mpnn/README_REPRODUCIBLE.md) - Complete guide
- [QUICK_REFERENCE.md](mpnn/QUICK_REFERENCE.md) - Parameter reference
- [config_example_qm9.py](mpnn/config_example_qm9.py) - Example with comments
