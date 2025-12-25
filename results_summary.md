# Results Summary

**Paper Revision for "Results in Engineering" Journal**

## Fix Applied
Fixed **oracle cheating bug** in `rl_env.py`: event flags now persist when sensors are OFF.

**Additional Improvement**: Trained on multiple seeds (10 traces) for better generalization.

---

## Synthetic Traces (16h simulation, 10 seeds)

| Policy | Detection (%) | Energy (mAh) | vs Always-On |
|--------|--------------|--------------|--------------|
| Always-On | 100.0 ± 0.0 | 250.0 ± 0.0 | baseline |
| Periodic-5/30 | 3.2 ± 0.2 | 27.8 ± 0.0 | -88.9% |
| Heuristic | 44.0 ± 1.1 | 73.1 ± 1.0 | -70.8% |
| **RL (Fixed)** | **82.4 ± 27.5** | 202.0 ± 67.3 | **-19.2%** |

---

## MIT-BIH Real Data (48 records)

| Policy | Detection (%) | Energy (mAh) |
|--------|--------------|--------------|
| Always-On | 95.8 ± 20.0 | 7.52 ± 0.0 |
| Heuristic | 64.4 ± 33.2 | 3.74 ± 2.5 |
| **RL (Fixed)** | **86.8 ± 19.4** | 7.14 ± 0.1 |

> **Key Result**: RL achieves **86.8% detection** on real ECG data vs 64.4% for Heuristic.
> Near Always-On performance with modest energy savings.

---

## Key Findings

1. **Methodological fix applied**: No oracle cheating - flags persist when sensors OFF
2. **Multi-seed training**: 5000 episodes across 10 different traces
3. **Strong real-data performance**: 86.8% detection on MIT-BIH
4. **RL outperforms Heuristic**: +22% detection improvement on real ECG data

---

## Files Produced

| File | Description |
|------|-------------|
| `rl_env.py` | Fixed with persistence logic |
| `q_table_fixed.pkl` | Retrained Q-table (5000 episodes, 10 seeds) |
| `q_table_fixed_convergence.png` | Training convergence plot |
| `evaluate_mitbih.py` | MIT-BIH evaluation script |
| `baselines.py` | Heuristic and baseline policies |
| `reproduce_results.ipynb` | Colab reproducibility notebook |
| `mitbih_summary.csv` | MIT-BIH evaluation results |
| `mitbih_results.csv` | Per-record MIT-BIH results |
| `synthetic_results.csv` | Synthetic evaluation results |
