"""
Aggressive Pareto Evaluation Script for Beta-Sweep Experiments.
Evaluates 3 aggressive beta configurations (0.05, 0.5, 1.0) on synthetic data.
"""
import sys
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.rl_env import HealthWearableEnv

SENSOR_COSTS = [10, 4, 1]

def load_q_table(path: Path) -> Dict:
    with path.open('rb') as f:
        return pickle.load(f)

def greedy_policy(Q: Dict, state: Tuple) -> int:
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))

def generate_synthetic_trace(n_steps: int = 12_000, seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [
        {'arr_flag': int(rng.random() < 0.10),
         'bp_flag': int(rng.random() < 0.30),
         'fever_flag': int(rng.random() < 0.10)}
        for _ in range(n_steps)
    ]

def evaluate_policy(data: List[Dict], policy_fn) -> Tuple[float, float]:
    """Evaluate policy, return (detection_rate, energy_mAh)."""
    env = HealthWearableEnv(
        data=data, sensor_costs=SENSOR_COSTS, max_time_steps=len(data)
    )
    
    state = env.reset()
    det_hits = det_total = energy = 0
    
    while not env.done:
        action = policy_fn(state)
        next_state, _, done, _ = env.step(action)
        
        ecg_on = (action >> 2) & 1
        ppg_on = (action >> 1) & 1
        tmp_on = action & 1
        energy += SENSOR_COSTS[0]*ecg_on + SENSOR_COSTS[1]*ppg_on + SENSOR_COSTS[2]*tmp_on
        
        if env.t <= len(data):
            gt = data[env.t - 1]
            for flag, on in [('arr_flag', ecg_on), ('bp_flag', ppg_on), ('fever_flag', tmp_on)]:
                if gt[flag]:
                    det_total += 1
                    if on: det_hits += 1
        
        if done: break
        state = next_state
    
    det_rate = (det_hits / det_total * 100) if det_total > 0 else 0.0
    energy_mAh = energy * 5 / 3600
    return det_rate, energy_mAh

def evaluate_synthetic(Q: Dict, n_seeds: int = 10) -> Tuple[float, float, float, float]:
    det_rates, energies = [], []
    for seed in range(n_seeds):
        data = generate_synthetic_trace(seed=seed)
        det, energy = evaluate_policy(data, lambda s: greedy_policy(Q, s))
        det_rates.append(det)
        energies.append(energy)
    return np.mean(det_rates), np.std(det_rates), np.mean(energies), np.std(energies)

def main():
    # Aggressive beta configurations
    configs = [
        ('Safety (0.05)', 'q_table_beta_0.05.pkl'),
        ('Balanced (0.5)', 'q_table_beta_0.5.pkl'),
        ('Saver (1.0)', 'q_table_beta_1.0.pkl'),
    ]
    
    results = []
    always_on_energy = 250.0
    
    print("="*70)
    print("AGGRESSIVE PARETO EVALUATION: Beta Sweep Results")
    print("="*70)
    
    for name, qtable_path in configs:
        path = Path(qtable_path)
        if not path.exists():
            print(f"  {name}: Q-table not found, skipping...")
            continue
        
        print(f"\nEvaluating {name}...")
        Q = load_q_table(path)
        det_mean, det_std, energy_mean, energy_std = evaluate_synthetic(Q, n_seeds=10)
        energy_savings = (1 - energy_mean / always_on_energy) * 100
        
        print(f"  Detection: {det_mean:.1f}% ± {det_std:.1f}%")
        print(f"  Energy: {energy_mean:.1f} mAh ({energy_savings:.1f}% savings)")
        
        results.append({
            'policy': name,
            'det_syn_mean': det_mean,
            'det_syn_std': det_std,
            'energy_syn_mean': energy_mean,
            'energy_syn_std': energy_std,
            'energy_savings': energy_savings,
        })
    
    # Print summary table
    print("\n" + "="*70)
    print("PARETO FRONTIER SUMMARY")
    print("="*70)
    print(f"{'Policy':<20} {'Detection':<15} {'Energy (mAh)':<15} {'Savings':<12}")
    print("-"*70)
    
    for r in results:
        det_str = f"{r['det_syn_mean']:.1f}% ± {r['det_syn_std']:.1f}%"
        energy_str = f"{r['energy_syn_mean']:.1f} ± {r['energy_syn_std']:.1f}"
        print(f"{r['policy']:<20} {det_str:<15} {energy_str:<15} {r['energy_savings']:.1f}%")
    
    print("="*70)
    
    # Save results
    with open('pareto_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nResults saved to pareto_results.csv")

if __name__ == "__main__":
    main()
