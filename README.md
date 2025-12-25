# Energy-Aware Sensing: RL Framework for TinyML

A general reinforcement learning framework for energy-aware multi-sensor systems, with applications to wearable health monitoring and other battery-constrained sensing scenarios.

## 🔧 Key Features

### Realistic Partial Observability
The framework implements **persistence logic** to ensure deployment-realistic training. When a sensor is OFF, the corresponding event flag retains its previous value, preventing access to future states and ensuring the agent operates under realistic partial observability constraints.

### Pareto Frontier: Detection vs Energy Trade-off

The β parameter controls the energy penalty weight, enabling tunable operating points:

| Config | β | Detection | Energy Savings |
|--------|---|-----------|----------------|
| Safety-First | 0.05 | **83%** | 19% |
| Balanced | 0.5 | **67%** | 68% |
| Energy-Saver | 1.0 | **33%** | 87% |

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
│   └── rl_env.py                     # Persistence logic implementation
├── scripts/
│   ├── train_q_learning.py          # Multi-seed Q-learning with --beta arg
│   ├── pareto_eval.py               # Pareto frontier evaluation
│   ├── evaluate_mitbih.py           # MIT-BIH real data evaluation
│   └── baselines.py                 # Clinical heuristic baseline policies
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

### Persistence Logic
When a sensor is OFF, the corresponding event flag retains its previous value, ensuring realistic partial observability:
```python
if sensor_is_on:
    flag = ground_truth  # Update from sensor
else:
    flag = previous_flag  # Persist (realistic partial observability)
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

### Synthetic Pareto Frontier (16h simulation, 10 seeds)

| Policy Configuration | Detection Coverage (%) | Energy Savings (%) |
|---------------------|------------------------|-------------------|
| Safety-First (β=0.05) | 83.1 ± 27.7 | 19.5 |
| Balanced (β=0.5) | 67.2 ± 22.4 | 68.5 |
| Energy-Saver (β=1.0) | 32.9 ± 11.0 | 87.4 |
| Clinical Heuristic | 44.0 ± 1.1 | 70.8 |

### MIT-BIH Real-Data Validation (48 records)

| Policy | Detection (%) | Energy (mAh) |
|--------|---------------|--------------|
| Always-On | 95.8 | 7.52 |
| RL Safety (β=0.05) | 92.0 | 7.15 |
| Clinical Heuristic | 64.4 | 3.74 |

Validated on 48 MIT-BIH Arrhythmia Database records with 92% detection coverage (Safety configuration), outperforming the clinical heuristic baseline by 28 percentage points.

## License

MIT

## Citation

If you use this framework, please cite:
```bibtex
@article{ellallam2025energy,
  title={Energy-Efficient On-Device Reinforcement Learning for Adaptive Multi-Sensor Scheduling in Resource-Constrained Edge Systems},
  author={El Allam, Oussama},
  journal={Under Review},
  year={2025}
}
```
