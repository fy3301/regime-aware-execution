import re
import numpy as np

file_path = "results_multiseed.txt"

# containers
data = {
    "blind": [],
    "regime_aware": [],
    "regime_conditioned": [],
    "twap": [],
    "regime_rule": []
}

current_strategy = None

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()

        # detect strategy sections
        if "blind across" in line:
            current_strategy = "blind"
        elif "PPO regime-aware" in line:
            current_strategy = "regime_aware"
        elif "Regime Conditioned across" in line:
            current_strategy = "regime_conditioned"

        # extract WAP
        wap_match = re.search(r"WAP[:=]\s*([0-9.]+)", line)
        comp_match = re.search(r"Comp(?:letion)?[:=]\s*([0-9.]+)", line)

        if wap_match and current_strategy:
            wap = float(wap_match.group(1))

            comp = None
            if comp_match:
                comp = float(comp_match.group(1))

            data[current_strategy].append((wap, comp))

# compute stats
for k, vals in data.items():
    if not vals:
        continue

    waps = [v[0] for v in vals if v[0] is not None]
    comps = [v[1] for v in vals if v[1] is not None]

    print(f"\n{k.upper()}")
    print(f"WAP mean: {np.mean(waps):.4f}")
    print(f"WAP std : {np.std(waps):.4f}")
    if comps:
        print(f"Comp    : {np.mean(comps):.4f}")