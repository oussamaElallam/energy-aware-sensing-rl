# Energy-Aware Sensing: RL Framework for TinyML

A general reinforcement learning framework for energy-aware multi-sensor systems, with applications to wearable health monitoring and other battery-constrained sensing scenarios.

## Repro steps

- Python 3.7+
- Install deps: `pip install -r requirements.txt`
- Train (example): `python scripts/train_q_learning.py --episodes 500`
- Evaluate (example): `python scripts/lambda_sweep.py --lambda_values 0 1 3`

Reproduce paper table/figure: A ready-to-open sweep CSV is committed at `results/lambda_sweep.csv`.

## Overview

This framework provides reusable components for developing RL-based energy management policies in resource-constrained sensing applications. The system balances detection accuracy with power consumption using Q-learning with configurable risk-aware rewards.

## Directory Structure

```
├── framework/                         # Reusable RL + TinyML components
│   ├── rl_env.py                     # Base environment classes
│   ├── prepare_tiny_dataset.py       # Dataset preparation utilities
│   └── convert_to_c_array.py         # Model conversion tools
├── scripts/                          # Training and evaluation scripts
│   ├── train_q_learning.py          # Q-learning with risk-aware rewards
│   ├── lambda_sweep.py               # Parameter sweep utility
│   ├── synthetic_evaluation.py       # Long-duration evaluation
│   └── *.py                         # Model training scripts
├── case_studies/
│   └── health_wearable/              # Wearable health monitoring example
│       ├── firmware/                 # ESP32-S3 Arduino implementation
│       └── hardware_setup.md         # Hardware wiring guide
├── results/                          # Training results and logs
└── tests/                           # Unit tests
```

## Requirements

### Framework
- Python 3.7+
- NumPy
- scikit-learn (for dataset utilities)

### Health Wearable Case Study
- Arduino IDE
- ESP32-S3 board support
- TensorFlow Lite for Microcontrollers
- Hardware sensors (see `case_studies/health_wearable/hardware_setup.md`)

## Quick Start

### 1. Framework Usage

Train a basic Q-learning policy:
```bash
python scripts/train_q_learning.py --episodes 1000
```

Train with risk-aware rewards:
```bash
python scripts/train_q_learning.py --lambda_risk 1.0 --episodes 1000
```

Run parameter sweep:
```bash
python scripts/lambda_sweep.py --lambda_values 0 0.5 1.0 3.0
```

Sweep results are saved to `results/lambda_sweep.csv` (also committed once for reviewers).

### 2. Health Wearable Case Study

1. Install Arduino IDE and ESP32-S3 board support
2. Install TensorFlow Lite for Microcontrollers library
3. Connect hardware per `case_studies/health_wearable/hardware_setup.md`
4. Configure risk penalty in firmware by modifying `LAMBDA_RISK` (default 0.0)
5. Upload `case_studies/health_wearable/firmware/firmware/main.ino`

### 3. Testing

Run the test suite:
```bash
pytest tests/
```

Run specific reward tests:
```bash
pytest tests/test_reward.py -v
```

### 4. Hardware-in-the-Loop Evaluation

Generate example sensor traces and run HIL replay:
```bash
python scripts/hil_replay_stub.py --create_example
python scripts/hil_replay_stub.py --csv_file example_traces.csv --output hil_results.json
```

## Framework Features

### Risk-Aware Rewards
The framework supports risk-aware reward functions that penalize missed critical events:
```
reward = α·detection_success - β·energy_cost - λ·missed_events
```

### Configurable Environments
- `EnergyAwareSensingEnv`: Base class for multi-sensor RL environments
- `HealthWearableEnv`: Specialized for wearable health monitoring
- Extensible to other sensing applications

### Training & Evaluation
- Q-learning with ε-greedy exploration
- Parameter sweeps for hyperparameter tuning
- Long-duration synthetic evaluations
- Model conversion utilities for embedded deployment

## Case Studies

### Health Wearable
Demonstrates energy-aware sensing for wearable health monitoring:
- **Sensors**: ECG, PPG, Temperature
- **Events**: Arrhythmia, blood pressure anomalies, fever
- **Platform**: ESP32-S3 with TinyML models
- **Results**: 16-hour evaluation showing power/accuracy trade-offs

## License

MIT

## Paper Results

All results reported in our paper are in `paper_results/`:
- `paper_results_lambdaA.json`: Lambda robustness analysis (λ=[0,20,50,100,200])
- `paper_tables.txt`: Formatted tables and statistics for the paper

Key findings:
- 55% energy reduction with 50% detection coverage
- Policy robust across λ∈[0,200]
- Statistical significance: p<0.002 vs baselines
