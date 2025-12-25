#!/usr/bin/env python3
"""
Lambda sweep script for risk-aware RL training.
Runs train_q_learning.py with different lambda_risk values and saves results.
"""

import subprocess
import csv
import os
from pathlib import Path
import numpy as np
import time

def run_lambda_sweep(lambda_values=[0, 1, 3], episodes=1000, output_dir="results", results_file=None):
    """Run Q-learning training with different lambda_risk values.

    If results_file is provided, write the CSV to that exact path; otherwise,
    write to <output_dir>/lambda_sweep.csv.
    """
    
    # Create results directory
    Path(output_dir).mkdir(exist_ok=True)
    
    results = []
    
    for lambda_risk in lambda_values:
        print(f"\n=== Training with lambda_risk = {lambda_risk} ===")
        
        # Output filename for this lambda value
        qtable_file = f"qtable_lambda_{lambda_risk}.csv"
        
        # Run training script
        cmd = [
            "python", "train_q_learning.py",
            "--lambda_risk", str(lambda_risk),
            "--episodes", str(episodes),
            "--output", qtable_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="scripts")
            
            if result.returncode == 0:
                # Parse output for metrics
                output_lines = result.stdout.strip().split('\n')
                lambda_line = [line for line in output_lines if "Lambda_risk:" in line]
                reward_line = [line for line in output_lines if "Avg reward" in line]
                sensor_line = [line for line in output_lines if "% of state" in line]
                
                lambda_val = float(lambda_line[0].split(":")[1].strip()) if lambda_line else lambda_risk
                avg_reward = float(reward_line[0].split(":")[1].strip()) if reward_line else 0.0
                sensor_pct = float(sensor_line[0].split("%")[0].strip()) if sensor_line else 0.0
                
                results.append({
                    'lambda_risk': lambda_val,
                    'avg_reward': avg_reward,
                    'sensor_on_percentage': sensor_pct,
                    'qtable_file': qtable_file,
                    'status': 'success'
                })
                
                print(f"✓ Success: avg_reward={avg_reward:.2f}, sensor_on={sensor_pct:.1f}%")
                
            else:
                print(f"✗ Failed: {result.stderr}")
                results.append({
                    'lambda_risk': lambda_risk,
                    'avg_reward': 0.0,
                    'sensor_on_percentage': 0.0,
                    'qtable_file': qtable_file,
                    'status': 'failed',
                    'error': result.stderr
                })
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            results.append({
                'lambda_risk': lambda_risk,
                'avg_reward': 0.0,
                'sensor_on_percentage': 0.0,
                'qtable_file': qtable_file,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results to CSV
    if results_file:
        results_path = Path(results_file)
        results_path.parent.mkdir(exist_ok=True)
    else:
        results_path = Path(output_dir) / "lambda_sweep.csv"
    fieldnames = ['lambda_risk', 'avg_reward', 'sensor_on_percentage', 'qtable_file', 'status']
    try:
        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({k: result[k] for k in fieldnames if k in result})
    except PermissionError:
        alt_path = results_path.with_name(f"{results_path.stem}_{int(time.time())}{results_path.suffix}")
        with open(alt_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({k: result[k] for k in fieldnames if k in result})
        results_path = alt_path
    
    print(f"\n=== Results saved to {results_path} ===")
    
    # Print summary
    print("\nSummary:")
    for result in results:
        if result['status'] == 'success':
            print(f"λ={result['lambda_risk']:3.1f}: reward={result['avg_reward']:6.2f}, sensors_on={result['sensor_on_percentage']:5.1f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run lambda sweep for risk-aware RL')
    parser.add_argument('--lambda_values', nargs='+', type=float, default=[0, 1, 3],
                       help='Lambda risk values to sweep (default: [0, 1, 3])')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per training run (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--out', type=str, default=None,
                        help='Optional path to write the results CSV (e.g., results/lambda_sweep.csv)')
    
    args = parser.parse_args()
    
    run_lambda_sweep(
        lambda_values=args.lambda_values,
        episodes=args.episodes,
        output_dir=args.output_dir,
        results_file=args.out,
    )
