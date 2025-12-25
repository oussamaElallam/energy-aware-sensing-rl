# Results Summary

**Paper Submission for "Results in Engineering" Journal**

## Key Feature: Persistence Logic
The framework implements **persistence logic** to ensure realistic partial observability. Event flags only update when the corresponding sensor is ON, preventing access to future states during training.

**Additional Enhancement**: Multi-seed training (10 traces) for improved generalization.

---

## Synthetic Pareto Frontier (16h simulation, 10 seeds)

| Policy Configuration | Detection Coverage (%) | Energy Savings (%) |
|---------------------|------------------------|-------------------|
| Safety-First (β=0.05) | 83.1 ± 27.7 | 19.5 |
| Balanced (β=0.5) | 67.2 ± 22.4 | 68.5 |
| Energy-Saver (β=1.0) | 32.9 ± 11.0 | 87.4 |
| Clinical Heuristic | 44.0 ± 1.1 | 70.8 |

---

## MIT-BIH Real-Data Validation (48 records)

| Policy | Detection (%) | Energy (mAh) |
|--------|---------------|--------------|
| Always-On | 95.8 ± 20.0 | 7.52 ± 0.0 |
| RL Safety (β=0.05) | 92.0 ± 19.4 | 7.15 ± 0.04 |
| Clinical Heuristic | 64.4 ± 33.2 | 3.74 ± 2.5 |

> **Key Result**: RL achieves **92% detection** on real ECG data, outperforming the clinical heuristic by 28 percentage points.

---

## Key Findings

1. **Realistic training**: Persistence logic ensures deployment-realistic partial observability
2. **Multi-seed training**: 5000 episodes across 10 different traces for generalization
3. **Strong real-data performance**: 92% detection on MIT-BIH
4. **RL outperforms baselines**: +28% detection improvement over clinical heuristic

---

## Files Produced

| File | Description |
|------|-------------|
| `rl_env.py` | Persistence logic implementation |
| `q_table_beta_*.pkl` | Trained Q-tables (5000 episodes, 10 seeds) |
| `evaluate_mitbih.py` | MIT-BIH evaluation script |
| `baselines.py` | Clinical heuristic policies |
| `reproduce_results.ipynb` | Colab reproducibility notebook |
| `mitbih_pareto.csv` | MIT-BIH evaluation results |
| `pareto_results.csv` | Synthetic evaluation results |
