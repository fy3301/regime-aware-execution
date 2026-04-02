# Does Regime Awareness Improve Reinforcement Learning for Optimal Trade Execution?

**Abstract** *(write last — 150 words)*

---

## 1. Introduction

*(~400 words — write this)*

Financial markets exhibit distinct behavioral regimes — periods of sustained 
upward trends, downward trends, and high volatility. A skilled trader behaves 
differently in each: aggressive execution in rising markets to avoid chasing 
price, patient limit-order execution in falling markets to capture favorable 
prices. Yet most reinforcement learning approaches to optimal execution treat 
the market as a single uniform environment, learning one policy for all 
conditions.

Amrouni et al. (2022) demonstrated that simple hand-coded regime-aware rules 
outperform regime-blind strategies in simulated limit order book markets, and 
called for further research into learned regime-aware policies. We directly 
answer this call.

**Our contributions:**
1. We implement and extend the CTMSTOU simulation environment from Amrouni 
   et al. (2022) for use with RL training
2. We train PPO agents with and without regime information in the state space
3. We find that regime conditioning in the state alone is insufficient — the 
   agent learns a near-binary policy that fails to match hand-coded rules
4. We provide a regime-by-regime breakdown revealing where each strategy 
   succeeds and fails

**[TODO: add 1 paragraph on paper structure]**

---

## 2. Background

### 2.1 The Optimal Execution Problem

*(~200 words)*

A trader must buy Q shares before deadline T. Trading too fast causes market 
impact — large orders move prices adversely. Trading too slowly risks the 
price moving away unfavorably. The classic Almgren-Chriss (2001) model solves 
this as a mean-variance optimization, producing a deterministic schedule. 
However, this model assumes stationary market conditions and cannot adapt to 
regime changes.

We measure execution quality using:
- **Implementation Shortfall (IS):** difference between decision price and 
  execution price
- **Weighted Average Price (WAP):** volume-weighted average execution price, 
  normalized to starting price
- **Completion rate:** fraction of target shares executed before deadline

### 2.2 Market Regimes

*(~200 words — write this)*

Define what regimes are. Cite Amrouni, Hamilton (1989) on Markov switching 
models. Explain bullish vs bearish distinction used in this work.

### 2.3 Reinforcement Learning for Execution

*(~200 words — write this)*

Frame execution as MDP. State, action, reward. Cite Nevmyvaka et al. (2006) 
as foundational RL execution work. Mention PPO briefly.

---

## 3. Method

### 3.1 Market Simulation: CTMSTOU

*(~300 words)*

We use the Continuous-Time Markov Switching Trending Ornstein-Uhlenbeck 
(CTMSTOU) process introduced by Amrouni et al. (2022) as our market simulator. 
This process switches between bullish and bearish regimes according to a 
continuous-time Markov chain, with each regime modeled as a trending 
Ornstein-Uhlenbeck process.

Formally, the fundamental value follows:

dX_t = θ_{s_t}(M_{s_t} - X_t)dt + σ_{s_t}dW_t

where s_t ∈ {0,1} is the regime state governed by a CTMC with transition 
rates calibrated to BTC/USD data (λ=2.90/day, ω=0.812/day).

**Parameters used:**
- θ = 0.00005 (mean reversion speed)
- σ = 50.0 (noise)  
- μ_bull = 0.5, μ_bear = -0.5 (trend slopes)
- Starting price: 100,000

Each simulated day = 82,800 seconds (23 hours).

### 3.2 Execution Environment

*(~250 words)*

We implement a custom Gymnasium environment wrapping the CTMSTOU simulator.

**State space (regime-aware agent, dim=5):**
- s_1: shares remaining (normalized 0-1)
- s_2: time remaining (normalized 0-1)
- s_3: current price (normalized to starting price)
- s_4: rolling price volatility (10-period std, normalized)
- s_5: current regime label (0=bull, 1=bear)

**State space (blind agent, dim=4):** same without s_5.

**Action space:** continuous value a ∈ [0,1] representing fraction of 
remaining shares to execute in current period (60-second periods).

**Reward:**
r_t = -cost_t × (qty_t / Q) + 0.5 × (qty_t / Q)

At episode end, incompletion penalty:
r_T += -20.0 × (shares_remaining / Q)  if shares_remaining > 0

### 3.3 Baselines

*(~150 words)*

We compare against three rule-based baselines from Amrouni et al.:

- **TWAP:** executes equal share quantity every 60 seconds using limit orders
- **Full MO:** executes equal quantity using market orders (always fills)
- **Regime Aware rule:** in bull regime, executes k×q shares via market order; 
  in bear regime, executes q shares via limit order across k price levels. 
  (k=10, following Amrouni et al.)

### 3.4 RL Agent: PPO

*(~150 words — write this)*

Describe PPO briefly. State hyperparameters: learning_rate=3e-4, 
n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, 
total_timesteps=500,000. MlpPolicy. 4 parallel envs.

---

## 4. Experiments and Results

### 4.1 Overall Execution Performance

*(~200 words)*

Table 1 shows overall WAP and completion across 100 test episodes.

| Strategy | WAP_norm | Completion |
|---|---|---|
| TWAP | 1.0277 | 0.850 |
| Full MO | 1.0278 | 1.000 |
| Regime Aware rule | **0.9949** | 0.997 |
| PPO blind | 1.0003 | 1.000 |
| PPO regime-aware | 1.0002 | 0.999 |

Both PPO agents dramatically outperform TWAP on completion (1.000 vs 0.850), 
demonstrating that RL successfully learns to fully execute the parent order. 
However, neither RL agent matches the regime-aware rule on WAP, which achieves 
0.9949 — actually buying below the starting price on average.

**[TODO: 2-3 sentences explaining why RL completion is better than TWAP]**

### 4.2 Regime-Stratified Performance

*(~250 words — this is your key result)*

Table 2 breaks down WAP by the dominant regime of each episode.

| Strategy | Bull WAP | Bear WAP | Bull n | Bear n |
|---|---|---|---|---|
| TWAP | 1.0844 | 0.9769 | 95 | 105 |
| Regime Aware rule | 1.0036 | 0.9559 | 156 | 44 |
| PPO blind | 1.0000 | 1.0002 | 95 | 105 |
| PPO regime-aware | 1.0000 | 1.0002 | 95 | 105 |

**[TODO: write analysis — key points to hit:]**
- TWAP suffers severely in bull markets (1.0844) — regime blindness costs most 
  when market is trending against you
- Regime Aware rule excels in bear markets (0.9559) — patient limit orders 
  capture falling prices
- PPO agents achieve similar WAP across both regimes — learned a 
  regime-independent steady execution policy
- Regime-aware and blind PPO perform identically here — state augmentation 
  alone insufficient

### 4.3 Regime Sensitivity Analysis

*(~200 words)*

To directly test whether the regime-aware agent uses the regime signal, we 
conduct a perturbation experiment: we flip the regime label in the observation 
and measure the change in predicted action.

| Seed | True regime | True action | Flipped action | Difference |
|---|---|---|---|---|
| 0 | Bull (0) | 0.9195 | 0.0000 | 0.9195 |
| 1 | Bull (0) | 0.9196 | 0.0000 | 0.9196 |
| 2 | Bull (0) | 0.9195 | 0.0000 | 0.9195 |

The regime-aware agent is highly sensitive to the regime label — flipping from 
bull to bear causes action to drop from ~0.92 to 0.00. This confirms the agent 
learned a near-binary policy: execute aggressively in bull regimes, do nothing 
in bear regimes.

This binary behavior explains the performance gap with the hand-coded rule. 
The rule executes patiently but actively in bear regimes (limit orders across 
price levels), while the RL agent learned that bear regime = abstain. 
Abstaining avoids paying above starting price but sacrifices the opportunity 
to buy cheaply in falling markets.

---

## 5. Discussion

*(~300 words — write this)*

**Key points to cover:**
- State augmentation is necessary but not sufficient for regime-aware behavior
- The agent needs regime-conditioned reward shaping to learn truly different 
  per-regime strategies
- The binary policy is a local optimum: safe (avoids overpaying) but suboptimal
- Implications: reward design matters more than state design for regime 
  exploitation

**Limitations:**
- Simplified fill model (fixed fill probabilities vs real LOB mechanics)
- Only two regimes tested
- CTMSTOU is cleaner than real markets — results may not transfer directly

---

## 6. Conclusion

*(~150 words — write this)*

We asked: can a PPO agent conditioned on market regime outperform hand-coded 
regime-aware rules for optimal execution? Our answer is nuanced. RL agents 
learn excellent completion rates, but regime conditioning via state augmentation 
alone is insufficient to match hand-coded rules on execution cost.

The regime-aware agent learns to use the regime signal — but learns the wrong 
behavior: complete abstention in bear markets rather than patient execution. 
This points to a clear direction for future work.

**Future work:**
- Regime-conditioned reward functions
- Separate policies per regime (mixture of experts)
- Hierarchical RL: high-level regime classifier + low-level executor
- Testing on real LOB data via ABIDES-Markets

---

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
- Amrouni, S., Moulin, A., & Balch, T. (2022). CTMSTOU driven markets.
- Hamilton, J.D. (1989). A new approach to the economic analysis of 
  nonstationary time series.
- Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for 
  optimized trade execution.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms.
- Saqur, R. (2024). What teaches robots to walk, teaches them to trade too.