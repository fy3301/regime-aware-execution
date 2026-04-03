# Regime-Aware Reinforcement Learning for Optimal Trade Execution

Empirical study examining whether market regime information improves 
reinforcement learning agents for optimal trade execution. Built on the 
CTMSTOU simulation environment from Amrouni et al. (2022) — JP Morgan AI 
Research.

## Research Question

Can a PPO agent conditioned on market regime outperform hand-coded 
regime-aware execution rules? And if regime information is available, 
does state augmentation alone suffice, or is reward conditioning necessary?

## Key Findings

- RL agents achieve near-perfect order completion (1.000) vs TWAP (0.850)
- Neither state augmentation nor reward conditioning matches the hand-coded 
  regime-aware rule on execution cost (WAP 1.0003 vs 0.9949)
- Regime sensitivity is highly initialization-dependent — different training 
  seeds produce qualitatively different policies
- Reward conditioning introduces instability (WAP std 0.0131) without 
  performance gains

## Results

| Strategy | WAP (mean ± std) | Completion |
|---|---|---|
| TWAP | 1.0277 ± 0.000 | 0.850 |
| Full Market Order | 1.0278 ± 0.000 | 1.000 |
| Regime Aware rule | **0.9949 ± 0.000** | 0.997 |
| PPO blind | 1.0003 ± 0.0000 | 1.000 |
| PPO state-aware | 1.0004 ± 0.0001 | 1.000 |
| PPO reward-conditioned | 1.0069 ± 0.0131 | 0.996 |

## Structure

├── src/
│   ├── ctmstou.py        # CTMSTOU market simulator
│   ├── environment.py    # Gymnasium execution environment
│   ├── baselines.py      # TWAP, Full MO, Regime-Aware rule
│   ├── train.py          # PPO training + evaluation
│   └── plot_results.py   # Figure generation
├── paper/
│   ├── draft.md          # Full paper draft
│   └── figures/          # All paper figures
└── README.md

## Setup
```bash
conda create -n regime-exec python=3.9
conda activate regime-exec
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3 gymnasium numpy matplotlib
```

## Reproducing Results
```bash
cd src
python baselines.py      # reproduce rule-based baselines
python train.py          # train all RL agents (2-3 hours on CPU)
python plot_results.py   # generate figures
```

## Built On

- Amrouni et al. (2022) — [CTMSTOU driven markets](https://arxiv.org/abs/2202.00941)
- Schulman et al. (2017) — Proximal Policy Optimization
- Stable Baselines 3

## Paper

Full paper draft available in `paper/draft.md`. 
Preprint forthcoming on arXiv.