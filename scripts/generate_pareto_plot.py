"""Generate Pareto Frontier Plot."""
import matplotlib.pyplot as plt
import numpy as np

# Results from pareto_results.csv
results = {
    'Always-On': {'det': 100.0, 'det_std': 0, 'energy': 250.0, 'energy_std': 0},
    'Safety (β=0.05)': {'det': 83.1, 'det_std': 27.7, 'energy': 201.3, 'energy_std': 67.1},
    'Balanced (β=0.5)': {'det': 67.2, 'det_std': 22.4, 'energy': 78.8, 'energy_std': 26.3},
    'Saver (β=1.0)': {'det': 32.9, 'det_std': 11.0, 'energy': 31.5, 'energy_std': 10.5},
}

fig, ax = plt.subplots(figsize=(10, 7))

colors = {'Always-On': '#e74c3c', 'Safety (β=0.05)': '#27ae60', 
          'Balanced (β=0.5)': '#3498db', 'Saver (β=1.0)': '#9b59b6'}
markers = {'Always-On': 's', 'Safety (β=0.05)': 'o', 'Balanced (β=0.5)': 'D', 'Saver (β=1.0)': '^'}

for name in ['Always-On', 'Safety (β=0.05)', 'Balanced (β=0.5)', 'Saver (β=1.0)']:
    r = results[name]
    ax.errorbar(r['energy'], r['det'], xerr=r['energy_std'], yerr=r['det_std'],
                fmt=markers[name], markersize=14, color=colors[name],
                label=name, capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1.5)

# Draw Pareto frontier line
pareto = [(results[n]['energy'], results[n]['det']) for n in 
          ['Saver (β=1.0)', 'Balanced (β=0.5)', 'Safety (β=0.05)', 'Always-On']]
xs, ys = zip(*pareto)
ax.plot(xs, ys, 'k--', alpha=0.4, lw=2, label='Pareto Frontier')

# Fill area under frontier
ax.fill_between(xs, ys, alpha=0.1, color='gray')

ax.set_xlabel('Energy Consumption (mAh)', fontsize=14, fontweight='bold')
ax.set_ylabel('Detection Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Detection vs Energy Trade-off (Pareto Frontier)', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)
ax.set_ylim(0, 110)

# Add annotations
ax.annotate('Max Detection\n(High Energy)', xy=(250, 100), xytext=(200, 85),
            fontsize=9, ha='center', alpha=0.7,
            arrowprops=dict(arrowstyle='->', alpha=0.5))
ax.annotate('Max Savings\n(Low Detection)', xy=(31.5, 32.9), xytext=(70, 20),
            fontsize=9, ha='center', alpha=0.7,
            arrowprops=dict(arrowstyle='->', alpha=0.5))

plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved pareto_frontier.png")
plt.show()
