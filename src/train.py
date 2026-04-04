import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment import ExecutionEnv

def make_env(regime_aware, reward_mode='standard'):
    """Factory for a single ExecutionEnv instance"""
    def _init():
        return ExecutionEnv(regime_aware=regime_aware, reward_mode=reward_mode)
    return _init

def make_monitored_env(regime_aware, reward_mode='standard'):
    """Factory for monitored env (SB3 compatible)"""
    def _init():
        env = ExecutionEnv(regime_aware=regime_aware, reward_mode=reward_mode)
        return Monitor(env)
    return _init

def train_agent(regime_aware, total_timesteps=500_000, seed=42, 
                reward_mode='standard', log_curves=False):
    """Train one PPO agent"""
    label = "regime_aware" if regime_aware else "blind"
    print(f"\nTraining {label} agent (seed={seed}, steps={total_timesteps})...")

    train_env = make_vec_env(
        make_monitored_env(regime_aware, reward_mode), 
        n_envs=4, seed=seed
    )

    model = PPO(
        "MlpPolicy", train_env, verbose=0, seed=seed,
        learning_rate=3e-4, n_steps=512, batch_size=64,
        n_epochs=10, gamma=0.99,
    )

    rewards_log = []

    if log_curves:
        from stable_baselines3.common.callbacks import BaseCallback

        class RewardLogger(BaseCallback):
            def __init__(self, eval_env, eval_freq=2_048, n_eval_episodes=20):
                super().__init__()
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes

            def _on_step(self):
                if self.num_timesteps % self.eval_freq == 0:
                    episode_rewards = []
                    for _ in range(self.n_eval_episodes):
                        obs, _ = self.eval_env.reset()
                        done = False
                        total_r = 0
                        while not done:
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, r, done, _, _ = self.eval_env.step(action)
                            total_r += r
                        episode_rewards.append(total_r)
                    rewards_log.append({
                        'timestep': self.num_timesteps,
                        'mean_reward': float(np.mean(episode_rewards))
                    })
                    print(f"  t={self.num_timesteps}: mean_reward={np.mean(episode_rewards):.4f}")
                return True

        eval_env = ExecutionEnv(regime_aware=regime_aware, reward_mode=reward_mode)
        callback = RewardLogger(eval_env, eval_freq=2_048, n_eval_episodes=20)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        model.learn(total_timesteps=total_timesteps)

    return model, rewards_log

def evaluate_agent(model, regime_aware, reward_mode='standard', n_episodes=100):
    """Evaluate trained agent over multiple episodes"""
    label = "regime_aware" if regime_aware else "blind"
    results = []

    for seed in range(n_episodes):
        env = ExecutionEnv(regime_aware=regime_aware, reward_mode=reward_mode)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
        r = env.get_results()
        if r:
            results.append(r)

    print(f"Valid episodes: {len(results)}/{n_episodes}")
    if not results:
        print("No valid results - agent executed nothing")
        return None, None

    avg_wap = np.mean([r['wap_norm'] for r in results])
    avg_comp = np.mean([r['completion'] for r in results])
    print(f"{label:20s} | WAP_norm: {avg_wap:.4f} | Completion: {avg_comp:.3f}")
    return avg_wap, avg_comp

def evaluate_by_regime(model, regime_aware, n_episodes=200):
    """Evaluate trained agent split by dominant regime"""
    label = "regime_aware" if regime_aware else "blind"
    bull_results, bear_results = [], []

    for seed in range(n_episodes):
        env = ExecutionEnv(regime_aware=regime_aware)
        obs, _ = env.reset(seed=seed)
        done = False
        regimes_seen = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            regimes_seen.append(int(obs[4]) if regime_aware else env.current_regime)
        r = env.get_results()
        if r:
            dominant = 0 if np.mean(regimes_seen) < 0.5 else 1
            (bull_results if dominant == 0 else bear_results).append(r)

    print(f"\n{label} — by regime:")
    if bull_results:
        print(f"  Bull: WAP={np.mean([r['wap_norm'] for r in bull_results]):.4f}, n={len(bull_results)}")
    if bear_results:
        print(f"  Bear: WAP={np.mean([r['wap_norm'] for r in bear_results]):.4f}, n={len(bear_results)}")

def train_and_evaluate_seeds(regime_aware, n_seeds=5, timesteps=500_000,
                              reward_mode='standard', log_curves=False):
    """Train and evaluate multiple seeds"""
    label = "regime_aware" if regime_aware else "blind"
    all_wap, all_comp, all_curves = [], [], []

    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds} — {label} ({timesteps} steps)")
        model, curves = train_agent(
            regime_aware=regime_aware, total_timesteps=timesteps,
            seed=seed, reward_mode=reward_mode, log_curves=log_curves
        )
        wap, comp = evaluate_agent(model, regime_aware=regime_aware, reward_mode=reward_mode)
        if wap:
            all_wap.append(wap)
            all_comp.append(comp)
        if curves is not None:
            all_curves.append(curves)

    print(f"\n{label} — {timesteps} steps — across {n_seeds} seeds:")
    print(f"  WAP:  {np.mean(all_wap):.4f} ± {np.std(all_wap):.4f}")
    print(f"  Comp: {np.mean(all_comp):.4f} ± {np.std(all_comp):.4f}")

    if all_curves:
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/curves_{label}_{timesteps}.json', 'w') as f:
            json.dump(all_curves, f)

    return all_wap, all_comp, all_curves

def train_regime_conditioned(n_seeds=5, timesteps=500_000):
    """Train regime-conditioned agent across multiple seeds"""
    all_wap, all_comp = [], []

    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds} — regime_conditioned")
        model, _ = train_agent(
            regime_aware=True, total_timesteps=timesteps,
            seed=seed, reward_mode='regime_conditioned'
        )
        model.save(f"models/ppo_conditioned_seed{seed}")

        results = []
        for ep in range(100):
            eval_env = ExecutionEnv(regime_aware=True,
                                   reward_mode='regime_conditioned')
            obs, _ = eval_env.reset(seed=ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
            r = eval_env.get_results()
            if r:
                results.append(r)

        if results:
            wap = np.mean([r['wap_norm'] for r in results])
            comp = np.mean([r['completion'] for r in results])
            all_wap.append(wap)
            all_comp.append(comp)
            print(f"  Seed {seed}: WAP={wap:.4f}, Comp={comp:.3f}")

    print(f"\nRegime Conditioned across {n_seeds} seeds:")
    print(f"  WAP:  {np.mean(all_wap):.4f} ± {np.std(all_wap):.4f}")
    print(f"  Comp: {np.mean(all_comp):.4f} ± {np.std(all_comp):.4f}")
    return all_wap, all_comp

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Hyperparameter sensitivity (timesteps)
    print("\n=== HYPERPARAMETER SENSITIVITY ===")
    for steps in [100_000, 200_000, 500_000]:
        train_and_evaluate_seeds(
            regime_aware=True, n_seeds=5, 
            timesteps=steps, log_curves=False
        )

    # Learning curves (500k with logging)
    print("\n=== LEARNING CURVES ===")
    _, _, aware_curves = train_and_evaluate_seeds(
        regime_aware=True, n_seeds=5,
        timesteps=500_000, log_curves=True
    )
    _, _, blind_curves = train_and_evaluate_seeds(
        regime_aware=False, n_seeds=5,
        timesteps=500_000, log_curves=True
    )

    print("\nDone. Run plot_results.py to generate all figures.")