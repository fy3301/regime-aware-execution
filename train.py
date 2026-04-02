import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import ExecutionEnv

def make_env(regime_aware):
    def _init():
        return ExecutionEnv(regime_aware=regime_aware)
    return _init

def train_agent(regime_aware, total_timesteps=200_000, seed=42):
    label = "regime_aware" if regime_aware else "blind"
    print(f"\nTraining {label} agent...")

    env = make_vec_env(make_env(regime_aware), n_envs=4, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(f"ppo_{label}")
    print(f"Saved ppo_{label}.zip")
    return model

def evaluate_agent(model, regime_aware, n_episodes=100):
    label = "regime_aware" if regime_aware else "blind"
    print(f"\nEvaluating {label} agent over {n_episodes} episodes...")
    results = []

    for seed in range(n_episodes):
        env = ExecutionEnv(regime_aware=regime_aware)
        obs, _ = env.reset(seed=seed)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

        r = env.get_results()
        if r:
            results.append(r)

    print(f"Valid episodes: {len(results)}/100")  # add this line
    if not results:
        print("No valid results - agent executed nothing")
        return None, None
        
    avg_wap = np.mean([r['wap_norm'] for r in results])
    avg_comp = np.mean([r['completion'] for r in results])
    print(f"{label:20s} | WAP_norm: {avg_wap:.4f} | Completion: {avg_comp:.3f}")
    return avg_wap, avg_comp


def evaluate_by_regime(model, regime_aware, n_episodes=200):
    """Break down performance by dominant regime of each episode"""
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
            if regime_aware:
                regimes_seen.append(int(obs[4]))
            else:
                regimes_seen.append(env.current_regime)

        r = env.get_results()
        if r:
            dominant = 0 if np.mean(regimes_seen) < 0.5 else 1
            if dominant == 0:
                bull_results.append(r)
            else:
                bear_results.append(r)

    print(f"\n{label} — by regime:")
    if bull_results:
        print(f"  Bull days: WAP={np.mean([r['wap_norm'] for r in bull_results]):.4f}, "
              f"n={len(bull_results)}")
    if bear_results:
        print(f"  Bear days: WAP={np.mean([r['wap_norm'] for r in bear_results]):.4f}, "
              f"n={len(bear_results)}")

if __name__ == "__main__":
    # train both agents
    model_aware = train_agent(regime_aware=True, total_timesteps=500_000)
    model_blind = train_agent(regime_aware=False, total_timesteps=500_000)

    # evaluate both
    print("\n=== RESULTS ===")
    print(f"{'Strategy':20s} | {'WAP_norm':10s} | {'Completion':10s}")
    print("-" * 45)

    # baselines for reference
    print(f"{'TWAP (rule)':20s} | {'1.0277':10s} | {'0.850':10s}")
    print(f"{'Regime Aware (rule)':20s} | {'0.9949':10s} | {'0.997':10s}")

    evaluate_agent(model_aware, regime_aware=True)
    evaluate_agent(model_blind, regime_aware=False)
        
    evaluate_by_regime(model_aware, regime_aware=True)
    evaluate_by_regime(model_blind, regime_aware=False)