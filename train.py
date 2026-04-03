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

def train_and_evaluate_seeds(regime_aware, n_seeds=5, timesteps=500_000):
    label = "regime_aware" if regime_aware else "blind"
    all_wap, all_comp = [], []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds} — {label}")
        model = train_agent(regime_aware=regime_aware, 
                          total_timesteps=timesteps, seed=seed)
        wap, comp = evaluate_agent(model, regime_aware=regime_aware)
        if wap:
            all_wap.append(wap)
            all_comp.append(comp)
    
    print(f"\n{label} across {n_seeds} seeds:")
    print(f"  WAP:  {np.mean(all_wap):.4f} ± {np.std(all_wap):.4f}")
    print(f"  Comp: {np.mean(all_comp):.4f} ± {np.std(all_comp):.4f}")
    return all_wap, all_comp

def train_regime_conditioned(n_seeds=5, timesteps=500_000):
    from environment import ExecutionEnv
    
    def make_conditioned_env():
        def _init():
            return ExecutionEnv(regime_aware=True, 
                              reward_mode='regime_conditioned')
        return _init
    
    all_wap, all_comp = [], []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds} — regime_conditioned")
        env = make_vec_env(make_conditioned_env(), n_envs=4, seed=seed)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                   learning_rate=3e-4, n_steps=512, batch_size=64,
                   n_epochs=10, gamma=0.99)
        model.learn(total_timesteps=timesteps)
        model.save(f"ppo_conditioned_seed{seed}")
        
        # evaluate
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
    print("=== MULTI-SEED EVALUATION ===")
    print("Running 5 seeds each.\n")
    
    aware_wap, aware_comp = train_and_evaluate_seeds(regime_aware=True, 
                                                      n_seeds=5, 
                                                      timesteps=500_000)
    blind_wap, blind_comp = train_and_evaluate_seeds(regime_aware=False, 
                                                     n_seeds=5, 
                                                     timesteps=500_000)
    
    print("\n=== FINAL COMPARISON ===")
    print(f"{'Strategy':25s} | {'WAP mean±std':20s} | {'Completion':10s}")
    print("-" * 60)
    print(f"{'TWAP (rule)':25s} | {'1.0277 ± 0.000':20s} | {'0.850':10s}")
    print(f"{'Regime Aware (rule)':25s} | {'0.9949 ± 0.000':20s} | {'0.997':10s}")
    print(f"{'PPO blind':25s} | "
          f"{np.mean(blind_wap):.4f} ± {np.std(blind_wap):.4f}    | "
          f"{np.mean(blind_comp):.3f}")
    print(f"{'PPO regime-aware':25s} | "
          f"{np.mean(aware_wap):.4f} ± {np.std(aware_wap):.4f}    | "
          f"{np.mean(aware_comp):.3f}")
    
    train_regime_conditioned(n_seeds=5, timesteps=500_000)