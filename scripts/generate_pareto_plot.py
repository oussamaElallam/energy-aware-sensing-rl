"""Generate Complete Pareto Frontier Plot with All Baselines."""
import matplotlib.pyplot as plt
import numpy as np

# Results from pareto_results.csv
results = {
    'Always-On': {'det': 100.0, 'det_std': 0, 'energy': 250.0, 'energy_std': 0},
    'Safety (b=0.05)': {'det': 83.1, 'det_std': 27.7, 'energy': 201.3, 'energy_std': 67.1},
    'Balanced (b=0.5)': {'det': 67.2, 'det_std': 22.4, 'energy': 78.8, 'energy_std': 26.3},
    'Saver (b=1.0)': {'det': 32.9, 'det_std': 11.0, 'energy': 31.5, 'energy_std': 10.5},
    'Heuristic': {'det': 44.0, 'det_std': 1.1, 'energy': 73.1, 'energy_std': 1.0},
    'Periodic-5/30': {'det': 3.2, 'det_std': 0.2, 'energy': 27.8, 'energy_std': 0.0},
}

fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    'Always-On': '#e74c3c', 
    'Safety (b=0.05)': '#27ae60', 
    'Balanced (b=0.5)': '#3498db', 
    'Saver (b=1.0)': '#9b59b6',
    'Heuristic': '#f39c12',
    'Periodic-5/30': '#95a5a6',
}
markers = {
    'Always-On': 's', 
    'Safety (b=0.05)': 'o', 
    'Balanced (b=0.5)': 'D', 
    'Saver (b=1.0)': '^',
    'Heuristic': 'p',
    'Periodic-5/30': 'X',
}

order = ['Always-On', 'Safety (b=0.05)', 'Balanced (b=0.5)', 'Saver (b=1.0)', 'Heuristic', 'Periodic-5/30']

for name in order:
    r = results[name]
    ax.errorbar(r['energy'], r['det'], xerr=r['energy_std'], yerr=r['det_std'],
                fmt=markers[name], markersize=14, color=colors[name],
                label=name, capsize=4, capthick=1.5, elinewidth=1.5, 
                markeredgecolor='white', markeredgewidth=1.5)

# Draw RL Pareto frontier
pareto = [(results[n]['energy'], results[n]['det']) for n in 
          ['Saver (b=1.0)', 'Balanced (b=0.5)', 'Safety (b=0.05)', 'Always-On']]
xs, ys = zip(*pareto)
ax.plot(xs, ys, 'k--', alpha=0.4, lw=2, label='RL Pareto Frontier')
ax.fill_between(xs, ys, alpha=0.08, color='green')

ax.set_xlabel('Energy Consumption (mAh)', fontsize=14, fontweight='bold')
ax.set_ylabel('Detection Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Detection vs Energy Trade-off: RL Policies vs Baselines', fontsize=16, fontweight='bold')
ax.legend(loc='center right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)
ax.set_ylim(0, 110)

# Annotations
ax.annotate('RL dominates Heuristic', xy=(73, 44), xytext=(130, 35),
            fontsize=10, ha='center', color='#f39c12', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

ax.annotate('Misses most events', xy=(28, 3.2), xytext=(55, 12),
            fontsize=9, ha='center', alpha=0.7,
            arrowprops=dict(arrowstyle='->', alpha=0.5))

plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved pareto_frontier.png")
