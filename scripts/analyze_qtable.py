"""Analyze Q-table to understand policy behavior."""
import pickle
import numpy as np
import sys
sys.path.insert(0, '.')
from framework.rl_env import HealthWearableEnv

with open('q_table_fixed.pkl', 'rb') as f:
    Q = pickle.load(f)

print('Q-TABLE ANALYSIS')
print('='*60)

action_names = {0: 'OFF', 1: 'Tmp', 2: 'PPG', 3: 'PPG+Tmp', 4: 'ECG', 5: 'ECG+Tmp', 6: 'ECG+PPG', 7: 'ALL'}

# Sample a few states
test_states = [
    (10, 0, 0, 0, 0),  # Full battery, t=0, no events
    (10, 0, 1, 0, 0),  # Full battery, t=0, arrhythmia
    (10, 0, 0, 1, 0),  # Full battery, t=0, BP event
    (10, 0, 1, 1, 1),  # Full battery, all events
]

for state in test_states:
    print(f'\nState: bat={state[0]}, t={state[1]}, arr={state[2]}, bp={state[3]}, fev={state[4]}')
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    best_a = np.argmax(qvals)
    print(f'  Best action: {action_names[best_a]} (Q={qvals[best_a]:.2f})')
    for a, q in enumerate(qvals):
        marker = ' <--' if a == best_a else ''
        print(f'    {action_names[a]:>8}: {q:>8.2f}{marker}')

# Count how many states have each action as optimal
print('\n' + '='*60)
print('OPTIMAL ACTION DISTRIBUTION ACROSS ALL Q-TABLE STATES')
print('='*60)
from collections import Counter
optimal_actions = []
for (state, action), qval in Q.items():
    all_qvals = [Q.get((state, a), 0.0) for a in range(8)]
    if action == np.argmax(all_qvals):
        optimal_actions.append(action)

counts = Counter(optimal_actions)
total = len(optimal_actions)
for a in sorted(counts.keys()):
    print(f'  {action_names[a]:>8}: {counts[a]:>5} states ({counts[a]/total*100:.1f}%)')
