# Energy-Aware Sensing: RL Framework for TinyML

A general reinforcement learning framework for energy-aware multi-sensor systems, with applications to wearable health monitoring and other battery-constrained sensing scenarios.

## 🔧 Paper Revision Updates

### Critical Fix: Oracle Bug
Fixed a methodological flaw where the agent could see ground-truth event flags even when sensors were OFF.

**Solution**: Implemented **persistence logic** - event flags only update when the corresponding sensor is ON. Otherwise, flags persist (stale values).

### Pareto Frontier: Detection vs Energy Trade-off

| Config | β | Detection | Energy Savings |
|--------|---|-----------|----------------|
| Safety-First | 0.05 | **83%** | 19% |
| Balanced | 0.5 | **67%** | 68% |
| Energy-Saver | 1.0 | **33%** | 87% |

The β parameter controls the energy penalty weight, enabling tunable trade-offs.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with different energy penalties (β values)
python scripts/train_q_learning.py --episodes 5000 --beta 0.05 --n_seeds 10 --output q_table_beta_0.05

# Evaluate Pareto frontier
python scripts/pareto_eval.py

# Run all tests
pytest tests/
```

## Directory Structure

```
├── framework/                         # Core RL environment
│   └── rl_env.py                     # Fixed persistence logic
├── scripts/
│   ├── train_q_learning.py          # Multi-seed Q-learning with --beta arg
│   ├── pareto_eval.py               # Pareto frontier evaluation
│   ├── evaluate_mitbih.py           # MIT-BIH real data evaluation
│   └── baselines.py                 # Heuristic baseline policies
├── tests/
│   ├── test_persistence.py          # Persistence logic tests (9 tests)
│   ├── test_rl_env.py               # Environment tests
│   └── test_reward.py               # Reward function tests
├── reproduce_results.ipynb           # Colab notebook with Pareto plot
├── pareto_results.csv                # Summary table
├── q_table_beta_*.pkl               # Trained Q-tables
└── mitbih_*.csv                      # MIT-BIH evaluation results
```

## Reproducibility

### Google Colab
Open `reproduce_results.ipynb` to reproduce all results including the Pareto frontier plot.

### MIT-BIH Evaluation
```bash
python scripts/evaluate_mitbih.py --qtable q_table_beta_0.05.pkl
```

Downloads 48 real ECG records from PhysioNet and evaluates detection performance.

## Requirements

- Python 3.7+
- NumPy
- matplotlib
- wfdb (for MIT-BIH evaluation)
- pytest (for testing)

## Framework Features

### Persistence Logic (Fixed)
When a sensor is OFF, the corresponding event flag retains its previous value:
```python
if sensor_is_on:
    flag = ground_truth  # Update from sensor
else:
    flag = previous_flag  # Persist (no oracle access)
```

### Configurable Energy Penalty
```bash
# High detection (low savings)
python scripts/train_q_learning.py --beta 0.05

# Balanced
python scripts/train_q_learning.py --beta 0.5

# Max savings (lower detection)
python scripts/train_q_learning.py --beta 1.0
```

### Multi-Seed Training
Train on multiple random traces for better generalization:
```bash
python scripts/train_q_learning.py --n_seeds 10 --episodes 5000
```

## Results Summary

### Synthetic Traces (16h simulation, 10 seeds)
| Policy | Detection | Energy (mAh) | vs Always-On |
|--------|-----------|--------------|--------------|
| Always-On | 100% | 250 | baseline |
| Safety (β=0.05) | 83% | 201 | -20% |
| Balanced (β=0.5) | 67% | 79 | -68% |
| Saver (β=1.0) | 33% | 32 | -87% |

### MIT-BIH Real ECG Data (48 records)
| Policy | Detection | Energy (mAh) |
|--------|-----------|--------------|
| Always-On | 96% | 7.5 |
| RL (β=0.05) | 87% | 7.1 |
| Heuristic | 64% | 3.7 |

## License

MIT

## Citation

If you use this framework, please cite:
```bibtex
@article{ellallam2024energy,
  title={Energy-Aware Sensing with Reinforcement Learning for TinyML},
  author={El Allam, Oussama},
  journal={Results in Engineering},
  year={2024}
}
```
