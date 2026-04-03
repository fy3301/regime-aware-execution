import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Figure 1: Strategy comparison bar chart ──────────────────
strategies = ['TWAP', 'Full MO', 'Regime\nAware Rule', 
              'PPO\nBlind', 'PPO State\nAware', 'PPO Reward\nConditioned']
wap_means  = [1.0277, 1.0278, 0.9949, 1.0003, 1.0004, 1.0069]
wap_stds   = [0.0,    0.0,    0.0,    0.0000, 0.0001, 0.0131]
completion = [0.850,  1.000,  0.997,  1.000,  1.000,  0.996]
colors = ['#4C72B0','#4C72B0','#2ca02c','#DD8452','#DD8452','#9467bd']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars = ax1.bar(strategies, wap_means, yerr=wap_stds, 
               color=colors, capsize=4, alpha=0.85, edgecolor='black')
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
            label='Starting price')
ax1.set_ylabel('WAP Normalized')
ax1.set_title('Execution Cost by Strategy\n(lower is better)')
ax1.legend()
ax1.set_ylim(0.97, 1.06)

# annotate bars
for bar, val in zip(bars, wap_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

bars2 = ax2.bar(strategies, completion, color=colors, 
                alpha=0.85, edgecolor='black')
ax2.set_ylabel('Completion Rate')
ax2.set_title('Order Completion by Strategy\n(higher is better)')
ax2.set_ylim(0.8, 1.05)

for bar, val in zip(bars2, completion):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('paper/fig1_strategy_comparison.png', dpi=200, 
            bbox_inches='tight')
print("Saved fig1")

# ── Figure 2: Regime-stratified WAP ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(4)
width = 0.35
strats = ['TWAP', 'Regime Aware Rule', 'PPO Blind', 'PPO Regime-Aware']
bull_wap = [1.0844, 1.0036, 1.0000, 1.0000]
bear_wap = [0.9769, 0.9559, 1.0002, 1.0002]

b1 = ax.bar(x - width/2, bull_wap, width, label='Bull days',
            color='#2ca02c', alpha=0.85, edgecolor='black')
b2 = ax.bar(x + width/2, bear_wap, width, label='Bear days',
            color='#d62728', alpha=0.85, edgecolor='black')

ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.4,
           label='Starting price')
ax.set_ylabel('WAP Normalized')
ax.set_title('Execution Cost by Strategy and Market Regime\n(lower is better)')
ax.set_xticks(x)
ax.set_xticklabels(strats)
ax.legend()
ax.set_ylim(0.93, 1.12)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('paper/fig2_regime_breakdown.png', dpi=200, bbox_inches='tight')
print("Saved fig2")

# ── Figure 3: Perturbation analysis ──────────────────────────
# use the regenerated data
seeds = [0, 1, 2, 3, 4]
true_actions  = [0.7384, 0.7383, 0.7384, 0.7385, 0.7384]
flip_actions  = [0.7788, 0.7788, 0.7788, 0.7789, 0.7788]
x = np.arange(len(seeds))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - width/2, true_actions, width, label='True regime (Bull=0)',
       color='#2ca02c', alpha=0.85, edgecolor='black')
ax.bar(x + width/2, flip_actions, width, label='Flipped regime (Bear=1)',
       color='#d62728', alpha=0.85, edgecolor='black')
ax.set_xlabel('Training seed')
ax.set_ylabel('Predicted action (fraction to execute)')
ax.set_title('Regime Sensitivity Across 5 Training Seeds\n'
             'Multi-seed agents show weak, inverted regime sensitivity')
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.legend()
ax.set_ylim(0.6, 0.9)
# add difference annotation
for i, (t, f) in enumerate(zip(true_actions, flip_actions)):
    ax.text(i, max(t,f) + 0.005, f'Δ={abs(t-f):.4f}', 
            ha='center', fontsize=7, color='purple')
plt.tight_layout()
plt.savefig('paper/fig3_perturbation.png', dpi=200, bbox_inches='tight')
print("Saved fig3")