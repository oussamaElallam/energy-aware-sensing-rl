# Unstructured Code Directory

This directory contains legacy and experimental code from early development stages. The code here is **not part of the main framework** and is kept for reference purposes only.

## Contents

### RL Agent training/
- **RL.py**: Legacy version of the RL environment (pre-framework refactor)
  - Does NOT include `lambda_risk` parameter
  - Superseded by `framework/rl_env.py`
- **run_policies.py**: Early policy evaluation script
  - Superseded by `scripts/synthetic_evaluation.py`

### Bp_estimator/
- Blood pressure estimation model training code
- Duplicate utility files (also in `framework/`)
- Kept for reference and reproducibility of early experiments

### ECG_Nano_Model/
- ECG classification model training code
- Early data preprocessing scripts
- Historical reference for model development

## Important Notes

⚠️ **Do not use this code for new development!**

- Use `framework/rl_env.py` for RL environments
- Use `scripts/train_q_learning.py` for training
- Use framework utilities instead of duplicates here

## Why Keep This Directory?

1. **Reproducibility**: Some early experimental results may reference this code
2. **Documentation**: Shows evolution of the framework
3. **Reference**: Contains working examples that may be useful for understanding design decisions

## Migration Status

The main framework (in `framework/` and `scripts/`) has been refactored to:
- ✅ Include risk-aware rewards (`lambda_risk`)
- ✅ Support configurable sensors and costs
- ✅ Proper code organization and testing
- ✅ Consistent parameter naming

If you need any of this functionality, refer to the main framework instead.
