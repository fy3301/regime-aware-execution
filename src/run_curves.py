import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import train_and_evaluate_seeds

os.makedirs('logs', exist_ok=True)

print("=== LEARNING CURVES ===")
_, _, aware_curves = train_and_evaluate_seeds(
    regime_aware=True, n_seeds=5,
    timesteps=500_000, log_curves=True
)
_, _, blind_curves = train_and_evaluate_seeds(
    regime_aware=False, n_seeds=5,
    timesteps=500_000, log_curves=True
)
print("Done. Run plot_results.py to generate fig4.")