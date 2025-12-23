"""Quick MIT-BIH evaluation for all beta configs."""
import sys
import csv
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.rl_env import HealthWearableEnv
import wfdb

SENSOR_COSTS = [10, 4, 1]
FS = 360
SAMPLES_PER_WINDOW = 5 * FS
ABNORMAL = ['L', 'R', 'A', 'V', 'F', '/', 'f', 'j', 'a', 'S', 'E']

def load_q(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def greedy(Q, s):
    return int(np.argmax([Q.get((s, a), 0.0) for a in range(8)]))

def convert(record_path):
    ann = wfdb.rdann(record_path, 'atr')
    rec = wfdb.rdrecord(record_path)
    n_win = rec.sig_len // SAMPLES_PER_WINDOW
    events = []
    for w in range(n_win):
        s0, s1 = w * SAMPLES_PER_WINDOW, (w+1) * SAMPLES_PER_WINDOW
        arr = any(s0 <= ann.sample[i] < s1 and ann.symbol[i] in ABNORMAL for i in range(len(ann.sample)))
        events.append({'arr_flag': int(arr), 'bp_flag': 0, 'fever_flag': 0})
    return events

def evaluate(events, policy_fn):
    env = HealthWearableEnv(data=events, sensor_costs=SENSOR_COSTS, max_time_steps=len(events))
    s = env.reset()
    hits = total = energy = 0
    while not env.done:
        a = policy_fn(s)
        s2, _, done, _ = env.step(a)
        ecg = (a >> 2) & 1
        energy += SENSOR_COSTS[0]*ecg + SENSOR_COSTS[1]*((a>>1)&1) + SENSOR_COSTS[2]*(a&1)
        if env.t <= len(events) and events[env.t-1]['arr_flag']:
            total += 1
            if ecg: hits += 1
        if done: break
        s = s2
    return (hits/total*100 if total else 0), energy * 5/3600

records = ['100','101','102','103','104','105','106','107','108','109',
           '111','112','113','114','115','116','117','118','119','121',
           '122','123','124','200','201','202','203','205','207','208',
           '209','210','212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

configs = [('Safety (beta=0.05)', 'q_table_beta_0.05.pkl'),
           ('Balanced (beta=0.5)', 'q_table_beta_0.5.pkl'),
           ('Saver (beta=1.0)', 'q_table_beta_1.0.pkl')]

results = []
for name, qtable in configs:
    print(f"\nEvaluating {name}...")
    Q = load_q(qtable)
    dets, enes = [], []
    for rid in records:
        try:
            events = convert(f"mitbih_data/{rid}")
            det, ene = evaluate(events, lambda s, Q=Q: greedy(Q, s))
            dets.append(det)
            enes.append(ene)
        except Exception as e:
            pass
    print(f"  Detection: {np.mean(dets):.1f}% +/- {np.std(dets):.1f}%")
    print(f"  Energy: {np.mean(enes):.2f} +/- {np.std(enes):.2f} mAh")
    results.append({'policy': name, 'det_mean': np.mean(dets), 'det_std': np.std(dets),
                    'energy_mean': np.mean(enes), 'energy_std': np.std(enes), 'n_records': len(dets)})

# Save
with open('mitbih_pareto.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print("\nSaved mitbih_pareto.csv")
