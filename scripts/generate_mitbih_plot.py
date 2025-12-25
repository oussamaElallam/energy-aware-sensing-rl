"""Generate MIT-BIH Pareto Frontier Plot."""
import matplotlib.pyplot as plt
import numpy as np

# MIT-BIH Results  
results = {
    'Always-On': {'det': 95.8, 'det_std': 20.0, 'energy': 7.52, 'energy_std': 0},
    'Safety (b=0.05)': {'det': 92.2, 'det_std': 19.4, 'energy': 7.15, 'energy_std': 0.04},
    'Heuristic': {'det': 64.4, 'det_std': 33.2, 'energy': 3.74, 'energy_std': 2.5},
    'Balanced (b=0.5)': {'det': 26.5, 'det_std': 17.1, 'energy': 3.46, 'energy_std': 0.79},
    'Saver (b=1.0)': {'det': 5.1, 'det_std': 9.9, 'energy': 1.18, 'energy_std': 0.5},
}

fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    'Always-On': '#e74c3c', 
    'Safety (b=0.05)': '#27ae60', 
    'Balanced (b=0.5)': '#3498db', 
    'Saver (b=1.0)': '#9b59b6',
    'Heuristic': '#f39c12',
}
markers = {
    'Always-On': 's', 
    'Safety (b=0.05)': 'o', 
    'Balanced (b=0.5)': 'D', 
    'Saver (b=1.0)': '^',
    'Heuristic': 'p',
}

order = ['Always-On', 'Safety (b=0.05)', 'Heuristic', 'Balanced (b=0.5)', 'Saver (b=1.0)']

for name in order:
    r = results[name]
    ax.errorbar(r['energy'], r['det'], xerr=r['energy_std'], yerr=r['det_std'],
                fmt=markers[name], markersize=14, color=colors[name],
                label=name, capsize=4, capthick=1.5, elinewidth=1.5, 
                markeredgecolor='white', markeredgewidth=1.5)

# RL Pareto frontier
pareto = [(results[n]['energy'], results[n]['det']) for n in 
          ['Saver (b=1.0)', 'Balanced (b=0.5)', 'Safety (b=0.05)', 'Always-On']]
xs, ys = zip(*pareto)
ax.plot(xs, ys, 'k--', alpha=0.4, lw=2, label='RL Pareto Frontier')
ax.fill_between(xs, ys, alpha=0.08, color='green')

ax.set_xlabel('Energy Consumption (mAh)', fontsize=14, fontweight='bold')
ax.set_ylabel('Detection Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('MIT-BIH Real ECG: Detection vs Energy Trade-off', fontsize=16, fontweight='bold')
ax.legend(loc='center right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 9)
ax.set_ylim(0, 110)

# Annotation for RL vs Heuristic
ax.annotate('Safety RL >> Heuristic\n(+28% detection)', 
            xy=(3.74, 64.4), xytext=(5.5, 55),
            fontsize=10, ha='center', color='#f39c12', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

plt.tight_layout()
plt.savefig('pareto_mitbih.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved pareto_mitbih.png")
