import numpy as np

# Historical monthly returns
# Columns:
# Stock, Bond, Gold

data = np.array([
    [0.05, 0.02, 0.03],
    [0.04, 0.01, 0.02],
    [-0.02, 0.03, 0.01],
    [0.06, 0.02, 0.04],
    [0.03, 0.01, 0.02],
    [0.05, 0.03, 0.03]
])

# Portfolio allocations
portfolios = {
    "Aggressive": [0.70, 0.20, 0.10],
    "Balanced": [0.50, 0.30, 0.20],
    "Conservative": [0.30, 0.50, 0.20]
}

print("Portfolio Performance")
print("---------------------")

results = {}

for name, weights in portfolios.items():

    weights = np.array(weights)

    # Calculate portfolio return
    portfolio_returns = data @ weights

    # Calculate cumulative return
    cumulative_return = (
        np.prod(1 + portfolio_returns) - 1
    )

    results[name] = cumulative_return

    print(
        name,
        ":",
        round(cumulative_return * 100, 2),
        "%"
    )

# Find best portfolio
best = max(
    results,
    key=results.get
)

print("\nBest Portfolio:", best)

print(
    "Highest Return:",
    round(results[best] * 100, 2),
    "%"
)