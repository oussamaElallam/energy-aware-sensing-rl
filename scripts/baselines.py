"""
Baseline Policies for Energy-Aware Sensing Evaluation

Contains:
- Always-On policy
- Periodic policy
- Heuristic policy (rule-based)
- Evaluation function for synthetic traces
"""

import sys
import csv
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.rl_env import HealthWearableEnv

# Sensor costs [ECG, PPG, Temp] in mA per 5s
SENSOR_COSTS = [10, 4, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Policy Implementations
# ─────────────────────────────────────────────────────────────────────────────

def always_on_policy(state: Tuple) -> int:
    """Always keep all sensors on. Action = 0b111 = 7"""
    return 0b111


def periodic_5_30_policy(state: Tuple) -> int:
    """
    Periodic policy: ECG ON for 5s every 30s.
    At 5s cadence, 30s = 6 time steps.
    Uses time_bucket from state.
    """
    _, time_bucket, *_ = state
    return 0b100 if (time_bucket % 6) == 0 else 0b000


def heuristic_policy(state: Tuple) -> int:
    """
    Rule-based policy for comparison:
    - Anomaly detected (in internal memory) → All ON for 30s
    - Otherwise → Periodic ECG every 30s, temp always ON
    
    This mirrors what a simple rule-based system might do.
    """
    battery, time_bucket, arr, bp, fever = state[:5]
    
    if arr or bp or fever:
        return 0b111  # All ON when anomaly detected
    elif time_bucket % 6 == 0:
        return 0b100  # ECG only (periodic check)
    else:
        return 0b001  # Temp only (always monitoring fever)


def load_q_table(path: Path) -> Dict:
    """Load Q-table from pickle file."""
    with path.open('rb') as f:
        return pickle.load(f)


def greedy_rl_policy(Q: Dict, state: Tuple) -> int:
    """Select action with highest Q-value."""
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Trace Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_trace(
    n_steps: int = 12_000,
    p_arr: float = 0.10,
    p_bp: float = 0.30,
    p_fever: float = 0.10,
    seed: int = 0,
) -> List[Dict]:
    """Generate synthetic event trace with specified prevalence."""
    rng = np.random.default_rng(seed)
    return [
        {
            "arr_flag": int(rng.random() < p_arr),
            "bp_flag": int(rng.random() < p_bp),
            "fever_flag": int(rng.random() < p_fever),
        }
        for _ in range(n_steps)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    data: List[Dict],
    policy_fn: Callable[[Tuple], int],
    lambda_risk: float = 0.0,
) -> Dict:
    """
    Evaluate a policy on a synthetic trace.
    
    Returns detection rate and energy consumption.
    """
    env = HealthWearableEnv(
        data=data,
        sensor_costs=SENSOR_COSTS,
        alpha=15.0,
        beta=0.008,
        lambda_risk=lambda_risk,
        max_battery=400_000,
        max_time_steps=len(data),
    )
    
    state = env.reset()
    
    det_hits = 0
    det_total = 0
    energy_cost = 0
    
    while not env.done:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)
        
        # Decode action
        ecg_on = (action >> 2) & 1
        ppg_on = (action >> 1) & 1
        tmp_on = action & 1
        
        # Energy bookkeeping
        energy_cost += (
            SENSOR_COSTS[0] * ecg_on +
            SENSOR_COSTS[1] * ppg_on +
            SENSOR_COSTS[2] * tmp_on
        )
        
        # Detection bookkeeping (using ground truth)
        if env.t <= len(data):
            gt = data[env.t - 1]
            for flag, on in [('arr_flag', ecg_on), ('bp_flag', ppg_on), ('fever_flag', tmp_on)]:
                if gt[flag]:
                    det_total += 1
                    if on:
                        det_hits += 1
        
        if done:
            break
        state = next_state
    
    detection_rate = (det_hits / det_total * 100) if det_total > 0 else 0.0
    energy_mAh = energy_cost * 5 / 3600  # mA·5s → mAh
    
    return {
        'detection_rate': detection_rate,
        'energy_mAh': energy_mAh,
        'det_hits': det_hits,
        'det_total': det_total,
    }


def run_synthetic_evaluation(
    q_table_path: Path = None,
    n_seeds: int = 10,
    n_steps: int = 12_000,
) -> Dict:
    """
    Run evaluation on synthetic traces with multiple seeds.
    
    Returns summary statistics for each policy.
    """
    # Load Q-table if provided
    Q = None
    if q_table_path and q_table_path.exists():
        print(f"Loading Q-table from {q_table_path}...")
        Q = load_q_table(q_table_path)
        print(f"  Loaded {len(Q)} entries")
    
    policies = [
        ("Always-On", always_on_policy),
        ("Periodic-5/30", periodic_5_30_policy),
        ("Heuristic", heuristic_policy),
    ]
    
    if Q is not None:
        policies.append(("RL-Fixed", lambda s: greedy_rl_policy(Q, s)))
    
    results = {name: {'det': [], 'energy': []} for name, _ in policies}
    
    print(f"\nRunning evaluation on {n_seeds} synthetic traces ({n_steps} steps each)...")
    
    for seed in range(n_seeds):
        data = generate_synthetic_trace(n_steps=n_steps, seed=seed)
        
        for name, policy_fn in policies:
            stats = evaluate_policy(data, policy_fn)
            results[name]['det'].append(stats['detection_rate'])
            results[name]['energy'].append(stats['energy_mAh'])
        
        if (seed + 1) % 5 == 0:
            print(f"  Completed {seed + 1}/{n_seeds} seeds")
    
    # Print summary
    print("\n" + "="*70)
    print("SYNTHETIC TRACES EVALUATION SUMMARY")
    print(f"({n_steps} steps × {n_seeds} seeds = {n_steps*5*n_seeds/3600:.1f} hours simulated)")
    print("="*70)
    print(f"{'Policy':<15} {'Detection (%)':<20} {'Energy (mAh)':<20} {'vs Always-On':<15}")
    print("-"*70)
    
    always_on_energy = np.mean(results["Always-On"]['energy'])
    
    summary = []
    for name, _ in policies:
        det_mean = np.mean(results[name]['det'])
        det_std = np.std(results[name]['det'])
        energy_mean = np.mean(results[name]['energy'])
        energy_std = np.std(results[name]['energy'])
        energy_reduction = (1 - energy_mean / always_on_energy) * 100 if always_on_energy > 0 else 0
        
        print(f"{name:<15} {det_mean:>6.1f} ± {det_std:<6.1f}     {energy_mean:>6.1f} ± {energy_std:<6.1f}     {energy_reduction:>+6.1f}%")
        
        summary.append({
            'policy': name,
            'det_mean': det_mean,
            'det_std': det_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_reduction': energy_reduction,
        })
    
    print("="*70)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate baseline policies on synthetic traces')
    parser.add_argument('--qtable', type=str, default='q_table_fixed.pkl',
                       help='Path to Q-table pickle file')
    parser.add_argument('--seeds', type=int, default=10,
                       help='Number of random seeds for evaluation')
    parser.add_argument('--steps', type=int, default=12_000,
                       help='Number of steps per trace (default: 12000 = 16h)')
    parser.add_argument('--output', type=str, default='synthetic_results.csv',
                       help='Output CSV file')
    args = parser.parse_args()
    
    q_path = Path(args.qtable) if args.qtable else None
    summary = run_synthetic_evaluation(
        q_table_path=q_path,
        n_seeds=args.seeds,
        n_steps=args.steps,
    )
    
    # Save results
    output_path = Path(args.output)
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResults saved to {output_path}")
