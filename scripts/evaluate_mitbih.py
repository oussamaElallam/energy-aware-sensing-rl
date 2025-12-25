"""
MIT-BIH Arrhythmia Database Evaluation Script

Downloads MIT-BIH data via wfdb, converts to event traces (5s windows),
and evaluates the fixed RL policy against Always-On and Heuristic baselines.

Note: MIT-BIH is ECG-only, so bp_flag and fever_flag are always 0.
"""

import sys
import csv
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.rl_env import HealthWearableEnv

# MIT-BIH abnormal beat symbols
ABNORMAL_SYMBOLS = ['L', 'R', 'A', 'V', 'F', '/', 'f', 'j', 'a', 'S', 'E']

# Sensor costs [ECG, PPG, Temp] in mA per 5s
SENSOR_COSTS = [10, 4, 1]

# MIT-BIH sampling frequency
FS_MITBIH = 360  # Hz

# Window parameters
WINDOW_SECONDS = 5
SAMPLES_PER_WINDOW = WINDOW_SECONDS * FS_MITBIH  # 1800 samples


def download_mitbih_record(record_id: str, data_dir: Path = Path("mitbih_data")):
    """Download a MIT-BIH record using wfdb."""
    import wfdb
    
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / record_id
    
    if not local_path.with_suffix('.dat').exists():
        print(f"  Downloading record {record_id}...")
        wfdb.dl_database('mitdb', str(data_dir), records=[record_id])
    
    return str(local_path)


def convert_record_to_events(record_path: str) -> List[Dict]:
    """
    Convert a MIT-BIH record to event trace format.
    
    Each 5-second window (1800 samples @ 360Hz) gets:
    - arr_flag = 1 if any abnormal beat in window
    - bp_flag = 0 (not available in MIT-BIH)
    - fever_flag = 0 (not available in MIT-BIH)
    """
    import wfdb
    
    # Read annotations
    ann = wfdb.rdann(record_path, 'atr')
    record = wfdb.rdrecord(record_path)
    
    total_samples = record.sig_len
    n_windows = total_samples // SAMPLES_PER_WINDOW
    
    events = []
    for w in range(n_windows):
        start_sample = w * SAMPLES_PER_WINDOW
        end_sample = (w + 1) * SAMPLES_PER_WINDOW
        
        # Check if any abnormal beat in this window
        arr_flag = 0
        for i, sample in enumerate(ann.sample):
            if start_sample <= sample < end_sample:
                if ann.symbol[i] in ABNORMAL_SYMBOLS:
                    arr_flag = 1
                    break
        
        events.append({
            'arr_flag': arr_flag,
            'bp_flag': 0,      # Not available in MIT-BIH
            'fever_flag': 0,   # Not available in MIT-BIH
        })
    
    return events


def load_q_table(path: Path) -> Dict:
    """Load Q-table from pickle file."""
    with path.open('rb') as f:
        return pickle.load(f)


def greedy_policy(Q: Dict, state: Tuple) -> int:
    """Select action with highest Q-value."""
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))


def always_on_policy(state: Tuple) -> int:
    """Always keep all sensors on."""
    return 0b111


def heuristic_policy(state: Tuple) -> int:
    """
    Rule-based policy:
    - Anomaly detected → All ON for 30s
    - Otherwise → Periodic ECG every 30s, temp always ON
    """
    battery, time_bucket, arr, bp, fever = state[:5]
    
    if arr or bp or fever:
        return 0b111  # All ON
    elif time_bucket % 6 == 0:
        return 0b100  # ECG only
    else:
        return 0b001  # Temp only


def evaluate_policy(env: HealthWearableEnv, policy_fn, data: List[Dict]) -> Dict:
    """
    Evaluate a policy on a single record.
    
    Returns detection stats and energy consumption.
    """
    state = env.reset()
    
    det_hits = 0
    det_total = 0
    energy_cost = 0
    
    while not env.done:
        action = policy_fn(state)
        next_state, _, done, info = env.step(action)
        
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
        
        # Detection bookkeeping (using ground truth from data)
        # Note: env.t is already advanced after step()
        if env.t <= len(data):
            gt = data[env.t - 1]  # Ground truth for this step
            if gt['arr_flag']:
                det_total += 1
                if ecg_on:
                    det_hits += 1
        
        if done:
            break
        state = next_state
    
    detection_rate = (det_hits / det_total * 100) if det_total > 0 else 0.0
    energy_mAh = energy_cost * WINDOW_SECONDS / 3600  # mA·s → mAh
    
    return {
        'det_hits': det_hits,
        'det_total': det_total,
        'detection_rate': detection_rate,
        'energy_mAh': energy_mAh,
    }


def evaluate_all_records(q_table_path: Path, output_dir: Path = Path(".")):
    """Evaluate policies on all 48 MIT-BIH records."""
    import wfdb
    
    # All MIT-BIH record IDs
    record_ids = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # Load Q-table
    print(f"Loading Q-table from {q_table_path}...")
    Q = load_q_table(q_table_path)
    print(f"  Loaded {len(Q)} Q-table entries")
    
    # Results storage
    results = []
    
    data_dir = output_dir / "mitbih_data"
    
    # Evaluate each record
    for record_id in record_ids:
        print(f"\nProcessing record {record_id}...")
        
        try:
            # Download and convert
            record_path = download_mitbih_record(record_id, data_dir)
            events = convert_record_to_events(record_path)
            print(f"  Converted to {len(events)} windows ({len(events)*5/60:.1f} min)")
            
            # Count anomalies
            n_anomalies = sum(1 for e in events if e['arr_flag'])
            print(f"  Anomaly windows: {n_anomalies} ({n_anomalies/len(events)*100:.1f}%)")
            
            # Skip if too short
            if len(events) < 10:
                print(f"  Skipping (too short)")
                continue
            
            # Evaluate each policy
            for policy_name, policy_fn in [
                ("Always-On", always_on_policy),
                ("Heuristic", heuristic_policy),
                ("RL-Fixed", lambda s: greedy_policy(Q, s)),
            ]:
                env = HealthWearableEnv(
                    data=events,
                    sensor_costs=SENSOR_COSTS,
                    alpha=15.0,
                    beta=0.008,
                    lambda_risk=0.0,
                    max_battery=400_000,
                    max_time_steps=len(events),
                )
                
                stats = evaluate_policy(env, policy_fn, events)
                
                results.append({
                    'record': record_id,
                    'policy': policy_name,
                    'det_hits': stats['det_hits'],
                    'det_total': stats['det_total'],
                    'detection_rate': stats['detection_rate'],
                    'energy_mAh': stats['energy_mAh'],
                })
                
                print(f"    {policy_name}: Det={stats['detection_rate']:.1f}%, Energy={stats['energy_mAh']:.2f} mAh")
        
        except Exception as e:
            print(f"  Error processing {record_id}: {e}")
            continue
    
    # Save detailed results
    results_csv = output_dir / "mitbih_results.csv"
    with results_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['record', 'policy', 'det_hits', 'det_total', 'detection_rate', 'energy_mAh'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {results_csv}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("MIT-BIH EVALUATION SUMMARY")
    print("="*60)
    
    for policy in ["Always-On", "Heuristic", "RL-Fixed"]:
        policy_results = [r for r in results if r['policy'] == policy]
        if policy_results:
            det_rates = [r['detection_rate'] for r in policy_results]
            energies = [r['energy_mAh'] for r in policy_results]
            
            print(f"\n{policy}:")
            print(f"  Detection: {np.mean(det_rates):.1f}% ± {np.std(det_rates):.1f}%")
            print(f"  Energy:    {np.mean(energies):.2f} ± {np.std(energies):.2f} mAh")
            print(f"  # Records: {len(policy_results)}")
    
    # Save summary
    summary_csv = output_dir / "mitbih_summary.csv"
    summary_data = []
    for policy in ["Always-On", "Heuristic", "RL-Fixed"]:
        policy_results = [r for r in results if r['policy'] == policy]
        if policy_results:
            det_rates = [r['detection_rate'] for r in policy_results]
            energies = [r['energy_mAh'] for r in policy_results]
            summary_data.append({
                'policy': policy,
                'det_mean': np.mean(det_rates),
                'det_std': np.std(det_rates),
                'energy_mean': np.mean(energies),
                'energy_std': np.std(energies),
                'n_records': len(policy_results),
            })
    
    with summary_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'det_mean', 'det_std', 'energy_mean', 'energy_std', 'n_records'])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\nSummary saved to {summary_csv}")
    
    return results, summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate policies on MIT-BIH data')
    parser.add_argument('--qtable', type=str, default='q_table_fixed.pkl',
                       help='Path to Q-table pickle file')
    parser.add_argument('--output', type=str, default='.',
                       help='Output directory for results')
    args = parser.parse_args()
    
    results, summary = evaluate_all_records(
        Path(args.qtable),
        Path(args.output)
    )
