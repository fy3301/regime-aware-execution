import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ctmstou import CTMSTOUFundamental

class ExecutionEnv(gym.Env):
    """
    RL environment for optimal trade execution in CTMSTOU markets.
    
    Agent must buy `total_shares` before time runs out.
    At each step it decides what fraction of remaining shares to execute.
    
    Two modes:
    - regime_aware=True:  agent sees regime label in state
    - regime_aware=False: agent does not see regime label
    """

    def __init__(self, regime_aware=True, total_shares=20000, 
                 total_steps=1380, seed=None, reward_mode='standard'):
        super().__init__()
        
        # 1380 steps = 23 hours / 60 seconds per step
        self.total_steps = total_steps
        self.total_shares = total_shares
        self.regime_aware = regime_aware
        self.init_seed = seed
        self.starting_price = 100000.0
        self.reward_mode = reward_mode

        # action: fraction of remaining shares to buy this step (0 to 1)
        self.action_space = spaces.Box(
            low=np.array([0.0]), 
            high=np.array([1.0]), 
            dtype=np.float32
        )

        # state: [shares_remaining, time_remaining, price_norm, volatility]
        # + [regime] if regime_aware
        obs_dim = 5 if regime_aware else 4
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # new random day each episode
        ep_seed = seed if seed is not None else np.random.randint(0, 100000)
        self.fund = CTMSTOUFundamental(seed=ep_seed)
        
        self.step_num = 0
        self.shares_remaining = float(self.total_shares)
        self.executed_value = 0.0
        self.executed_shares = 0.0
        self.price_history = []
        
        # advance simulator one step to get initial state
        price, regime = self.fund.step()
        self.current_price = price
        self.current_regime = regime
        self.price_history.append(price)

        return self._get_obs(), {}

    def _get_obs(self):
        shares_norm = self.shares_remaining / self.total_shares
        time_norm = 1.0 - (self.step_num / self.total_steps)
        price_norm = self.current_price / self.starting_price
        
        # rolling volatility: std of last 10 prices, normalized
        if len(self.price_history) >= 2:
            vol = np.std(self.price_history[-10:]) / self.starting_price
        else:
            vol = 0.0

        if self.regime_aware:
            return np.array([shares_norm, time_norm, price_norm, 
                           vol, float(self.current_regime)], dtype=np.float32)
        else:
            return np.array([shares_norm, time_norm, price_norm, 
                           vol], dtype=np.float32)

    def step(self, action):
        fraction = float(np.clip(action[0], 0.0, 1.0))
        qty = fraction * self.shares_remaining

        # advance market by 60 seconds (one period)
        prices_this_step = []
        for _ in range(60):
            price, regime = self.fund.step()
            prices_this_step.append(price)
            self.price_history.append(price)
        
        self.current_price = prices_this_step[-1]
        self.current_regime = regime
        avg_price = np.mean(prices_this_step)

        # execute at average price of the period
        if qty > 0:
            self.executed_value += qty * avg_price
            self.executed_shares += qty
            self.shares_remaining -= qty

        self.step_num += 1
        done = self.step_num >= self.total_steps
        
        # reward: negative execution cost, normalized
        if qty > 0:
            cost = (avg_price - self.starting_price) / self.starting_price
            
            if self.reward_mode == 'regime_conditioned':
                if self.current_regime == 0:  # bull — urgency matters
                    # penalize delay: reward fast execution
                    urgency_bonus = 0.3 * (qty / self.total_shares)
                    cost_penalty = -2.0 * cost * (qty / self.total_shares)
                    reward = cost_penalty + urgency_bonus
                else:  # bear — patience pays
                    # reward buying below starting price
                    savings = max(0, -cost)  # positive when price < starting
                    patience_reward = 1.0 * savings * (qty / self.total_shares)
                    progress_reward = 0.3 * (qty / self.total_shares)
                    reward = patience_reward + progress_reward
            else:
                # original reward
                cost_penalty = -cost * (qty / self.total_shares)
                progress_reward = 0.5 * (qty / self.total_shares)
                reward = cost_penalty + progress_reward
        else:
            reward = -0.001

        if done and self.shares_remaining > 0:
            incompletion_penalty = -20.0 * (self.shares_remaining / self.total_shares)
            reward += incompletion_penalty

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def get_results(self):
        if self.executed_shares == 0:
            return None
        wap = self.executed_value / self.executed_shares
        return {
            'wap': wap,
            'wap_norm': wap / self.starting_price,
            'completion': self.executed_shares / self.total_shares
        }