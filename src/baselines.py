import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ctmstou import CTMSTOUFundamental

def simulate_day(strategy, seed, total_shares=20000, total_seconds=82800):
    """
    Simulate one trading day using a baseline strategy.
    """
    fund = CTMSTOUFundamental(seed=seed)
    shares_remaining = total_shares
    executed_value = 0.0
    executed_shares = 0

    period = 60
    q = (total_shares / total_seconds) * period
    k = 10

    for t in range(total_seconds):
        price, regime = fund.step()
        if shares_remaining <= 0:
            break
        if t % period != 0:
            continue

        if strategy == 'twap':
            qty = min(q, shares_remaining)
            fill_prob = 0.85
            order_type = 'limit'

        elif strategy == 'regime_aware_1':
            if regime == 0:
                qty = min(k * q, shares_remaining)
                fill_prob = 1.0
                order_type = 'market'
            else:
                qty = min(q, shares_remaining)
                fill_prob = 0.80
                order_type = 'limit'

        elif strategy == 'full_mo':
            qty = min(q, shares_remaining)
            fill_prob = 1.0
            order_type = 'market'

        if order_type == 'market' or np.random.random() < fill_prob:
            executed_value += qty * price
            executed_shares += qty
            shares_remaining -= qty

    if executed_shares == 0:
        return None

    wap = executed_value / executed_shares
    completion = executed_shares / total_shares
    wap_normalized = wap / 100000
    return {'wap': wap, 'wap_norm': wap_normalized, 'completion': completion}


# run 100 days per strategy
n_days = 100
strategies = ['twap', 'regime_aware_1', 'full_mo']

for strat in strategies:
    np.random.seed(0)
    results = [simulate_day(strat, seed=i) for i in range(n_days)]
    results = [r for r in results if r is not None]
    avg_wap = np.mean([r['wap_norm'] for r in results])
    avg_comp = np.mean([r['completion'] for r in results])
    print(f"{strat:20s} | WAP_norm: {avg_wap:.4f} | Completion: {avg_comp:.3f}")


def simulate_day_detailed(strategy, seed, total_shares=20000, total_seconds=82800):
    """
    Simulate one day and track dominant regime for breakdown.
    """
    fund = CTMSTOUFundamental(seed=seed)
    shares_remaining = total_shares
    executed_value = 0.0
    executed_shares = 0
    period = 60
    q = (total_shares / total_seconds) * period
    k = 10
    regimes_seen = []

    for t in range(total_seconds):
        price, regime = fund.step()
        regimes_seen.append(regime)
        if shares_remaining <= 0:
            break
        if t % period != 0:
            continue
        if strategy == 'regime_aware_1':
            if regime == 0:
                qty = min(k * q, shares_remaining)
                fill_prob = 1.0
                order_type = 'market'
            else:
                qty = min(q, shares_remaining)
                fill_prob = 0.80
                order_type = 'limit'
        elif strategy == 'twap':
            qty = min(q, shares_remaining)
            fill_prob = 0.85
            order_type = 'limit'

        if order_type == 'market' or np.random.random() < fill_prob:
            executed_value += qty * price
            executed_shares += qty
            shares_remaining -= qty

    if executed_shares == 0:
        return None

    dominant_regime = 0 if np.mean(regimes_seen) < 0.5 else 1
    return {
        'wap_norm': executed_value / executed_shares / 100000,
        'completion': executed_shares / total_shares,
        'dominant_regime': dominant_regime
    }


# regime breakdown
for strat in ['twap', 'regime_aware_1']:
    np.random.seed(0)
    results = [simulate_day_detailed(strat, seed=i) for i in range(200)]
    results = [r for r in results if r]
    bull = [r for r in results if r['dominant_regime'] == 0]
    bear = [r for r in results if r['dominant_regime'] == 1]
    print(f"\n{strat}:")
    print(f"  Bull days (n={len(bull)}): WAP={np.mean([r['wap_norm'] for r in bull]):.4f}")
    print(f"  Bear days (n={len(bear)}): WAP={np.mean([r['wap_norm'] for r in bear]):.4f}")