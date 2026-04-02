import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CTMSTOUFundamental:
    def __init__(self, x0=100000, theta=0.00005, sigma=50.0,
                mu_bull=0.5, mu_bear=-0.5,
                lambda_=2.90, omega=0.812, dt=1.0, seed=None):
        self.x = x0
        self.M = x0
        self.theta = theta
        self.sigma = sigma
        self.mu = [mu_bull, mu_bear]
        self.lambda_ = lambda_ / 86400
        self.omega = omega / 86400
        self.dt = dt
        self.regime = 0
        self.rng = np.random.default_rng(seed)
        self.time_to_switch = self.rng.exponential(86400.0 / lambda_)
        self.elapsed = 0.0

    def step(self):
        self.elapsed += self.dt
        if self.elapsed >= self.time_to_switch:
            self.regime = 1 - self.regime
            self.elapsed = 0.0
            self.time_to_switch = self.rng.exponential(1.0 / self.omega)
        self.M += self.mu[self.regime] * self.dt
        dW = self.rng.normal(0, np.sqrt(self.dt))
        self.x += self.theta * (self.M - self.x) * self.dt + self.sigma * dW
        return self.x, self.regime

fund = CTMSTOUFundamental(seed=42)
prices, regimes = [], []
for _ in range(82800):
    p, r = fund.step()
    prices.append(p)
    regimes.append(r)

prices = np.array(prices)
regimes = np.array(regimes)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for i in range(len(prices) - 1):
    color = 'green' if regimes[i] == 0 else 'red'
    ax1.plot([i, i+1], [prices[i], prices[i+1]], color=color, linewidth=0.5)

ax1.set_title('CTMSTOU Price (green=bullish, red=bearish)')
ax1.set_ylabel('Price')

ax2.fill_between(range(len(regimes)), regimes, alpha=0.6, color='orange')
ax2.set_title('Regime (0=Bull, 1=Bear)')
ax2.set_ylabel('Regime')
ax2.set_xlabel('Time (seconds)')

switches = np.sum(np.diff(regimes) != 0)
fig.suptitle(f'CTMSTOU Simulation — {switches} regime switches in one day', fontsize=13)

plt.tight_layout()
plt.savefig('ctmstou_test.png', dpi=150)
print(f"Done. Switches: {switches}, Final price: {prices[-1]:.2f}")
print(f"Bull time: {np.mean(regimes==0)*100:.1f}%, Bear time: {np.mean(regimes==1)*100:.1f}%")
print("Plot saved to ctmstou_test.png")