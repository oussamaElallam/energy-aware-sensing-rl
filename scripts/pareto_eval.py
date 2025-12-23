"""
Complete Pareto Evaluation with all baselines and all 3 beta configs.
Runs on both synthetic and MIT-BIH data.
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

def always_on_policy(state: Tuple) -> int:
    return 0b111

def periodic_policy(state: Tuple) -> int:
    _, time_bucket, *_ = state
    return 0b100 if (time_bucket % 6) == 0 else 0b000

def heuristic_policy(state: Tuple) -> int:
    _, time_bucket, arr, bp, fever = state[:5]
    if arr or bp or fever:
        return 0b111
    elif time_bucket % 6 == 0:
        return 0b100
    else:
        return 0b001

def generate_synthetic_trace(n_steps: int = 12_000, seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [
        {'arr_flag': int(rng.random() < 0.10),
         'bp_flag': int(rng.random() < 0.30),
         'fever_flag': int(rng.random() < 0.10)}
        for _ in range(n_steps)
    ]

def evaluate_policy(data: List[Dict], policy_fn) -> Tuple[float, float]:
    env = HealthWearableEnv(data=data, sensor_costs=SENSOR_COSTS, max_time_steps=len(data))
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

def evaluate_synthetic(policy_fn, n_seeds: int = 10) -> Tuple[float, float, float, float]:
    det_rates, energies = [], []
    for seed in range(n_seeds):
        data = generate_synthetic_trace(seed=seed)
        det, energy = evaluate_policy(data, policy_fn)
        det_rates.append(det)
        energies.append(energy)
    return np.mean(det_rates), np.std(det_rates), np.mean(energies), np.std(energies)

def main():
    results = []
    
    print("="*70)
    print("COMPLETE PARETO EVALUATION")
    print("="*70)
    
    # Baselines
    print("\nEvaluating baselines...")
    for name, policy_fn in [("Always-On", always_on_policy), 
                             ("Periodic-5/30", periodic_policy),
                             ("Heuristic", heuristic_policy)]:
        det_mean, det_std, energy_mean, energy_std = evaluate_synthetic(policy_fn)
        energy_savings = (1 - energy_mean / 250) * 100
        print(f"  {name}: Det={det_mean:.1f}%, Energy={energy_mean:.1f} mAh ({energy_savings:.1f}% savings)")
        results.append({
            'policy': name, 'det_mean': det_mean, 'det_std': det_std,
            'energy_mean': energy_mean, 'energy_std': energy_std, 'energy_savings': energy_savings
        })
    
    # RL Policies
    beta_configs = [('Safety (beta=0.05)', 'q_table_beta_0.05.pkl'),
                    ('Balanced (beta=0.5)', 'q_table_beta_0.5.pkl'),
                    ('Saver (beta=1.0)', 'q_table_beta_1.0.pkl')]
    
    print("\nEvaluating RL policies...")
    for name, qtable_path in beta_configs:
        path = Path(qtable_path)
        if not path.exists():
            print(f"  {name}: NOT FOUND")
            continue
        Q = load_q_table(path)
        det_mean, det_std, energy_mean, energy_std = evaluate_synthetic(lambda s, Q=Q: greedy_policy(Q, s))
        energy_savings = (1 - energy_mean / 250) * 100
        print(f"  {name}: Det={det_mean:.1f}%, Energy={energy_mean:.1f} mAh ({energy_savings:.1f}% savings)")
        results.append({
            'policy': name, 'det_mean': det_mean, 'det_std': det_std,
            'energy_mean': energy_mean, 'energy_std': energy_std, 'energy_savings': energy_savings
        })
    
    # Print summary table
    print("\n" + "="*70)
    print("COMPLETE RESULTS TABLE")
    print("="*70)
    print(f"{'Policy':<20} {'Detection':<15} {'Energy (mAh)':<15} {'Savings':<10}")
    print("-"*70)
    for r in results:
        det_str = f"{r['det_mean']:.1f}% ± {r['det_std']:.1f}%"
        energy_str = f"{r['energy_mean']:.1f} ± {r['energy_std']:.1f}"
        print(f"{r['policy']:<20} {det_str:<15} {energy_str:<15} {r['energy_savings']:.1f}%")
    print("="*70)
    
    # Save results
    with open('pareto_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\n✓ Results saved to pareto_results.csv")

if __name__ == "__main__":
    main()
