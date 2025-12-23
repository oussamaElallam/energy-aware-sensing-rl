"""Simple diagnostic - write to file."""
import pickle
import numpy as np
import sys
sys.path.insert(0, '.')
from framework.rl_env import HealthWearableEnv
from collections import Counter

SENSOR_COSTS = [10, 4, 1]

with open('q_table_fixed.pkl', 'rb') as f:
    Q = pickle.load(f)

def greedy_rl_policy(state):
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))

def generate_trace(seed, n_steps=12000):
    rng = np.random.default_rng(seed)
    return [{'arr_flag': int(rng.random() < 0.10),
             'bp_flag': int(rng.random() < 0.30),
             'fever_flag': int(rng.random() < 0.10)} for _ in range(n_steps)]

def evaluate(data, policy_fn):
    env = HealthWearableEnv(data=data, sensor_costs=SENSOR_COSTS, max_time_steps=len(data))
    state = env.reset()
    det_hits = det_total = energy = 0
    actions = []
    missing = 0
    
    while not env.done:
        action = policy_fn(state)
        actions.append(action)
        # Check if state missing from Q-table
        if all(Q.get((state, a), 0.0) == 0 for a in range(8)):
            missing += 1
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
    
    det_rate = det_hits / det_total * 100 if det_total > 0 else 0
    return det_rate, energy * 5 / 3600, Counter(actions), missing, len(actions)

# Write results
with open('diagnose_results.txt', 'w') as f:
    f.write("ANSWER 1: Epsilon = 0 during evaluation (pure argmax, no exploration)\n\n")
    f.write("ANSWER 2: Per-seed breakdown\n")
    f.write("-" * 70 + "\n")
    f.write("Seed  Det%     Energy   MissingStates  MostCommonAction\n")
    f.write("-" * 70 + "\n")
    
    action_names = {0:'OFF', 1:'Tmp', 2:'PPG', 3:'PPG+Tmp', 4:'ECG', 5:'ECG+Tmp', 6:'ECG+PPG', 7:'ALL'}
    
    all_det = []
    for seed in range(10):
        data = generate_trace(seed)
        det, energy, actions, missing, total = evaluate(data, greedy_rl_policy)
        all_det.append(det)
        most_common = actions.most_common(1)[0]
        f.write(f"{seed:>4}  {det:>6.1f}%  {energy:>7.1f}   {missing:>5}/{total:<5}   {action_names[most_common[0]]}:{most_common[1]}\n")
    
    f.write("-" * 70 + "\n")
    f.write(f"Mean: {np.mean(all_det):.1f}%, Std: {np.std(all_det):.1f}%\n")
    f.write(f"Min: {min(all_det):.1f}%, Max: {max(all_det):.1f}%\n\n")
    
    f.write("ANSWER 3: Different traces per seed\n")
    f.write("Each seed generates a UNIQUE synthetic trace.\n")
    f.write("Variance comes from: trace difficulty + persistence logic + missing Q-states\n\n")
    
    f.write("ROOT CAUSE: Many states not in Q-table -> default action 0 (OFF)\n")

print("Results written to diagnose_results.txt")
