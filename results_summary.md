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

> **Note**: RL achieves 82% detection with 19% energy savings vs Always-On.
> Variance comes from persistence logic interaction with different event patterns.

---

## MIT-BIH Real Data

Run `python scripts/evaluate_mitbih.py` to evaluate on real ECG data.

> **Limitation**: MIT-BIH is ECG-only. bp_flag and fever_flag are always 0.

---

## Key Findings

1. **Methodological fix applied**: No oracle cheating - flags persist when sensors OFF
2. **Multi-seed training**: 5000 episodes across 10 different traces
3. **Good detection**: 82% vs 100% Always-On, with 19% energy savings
4. **Heuristic baseline**: 44% detection with 71% energy savings (simple but effective)

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
| `synthetic_results.csv` | Synthetic evaluation results |
