import numpy as np
import random
import math
true_ctr = [0.20, 0.50, 0.35]
n = 1000
epsilon = 0.1
def run_epsilon_greedy():
    clicks = [0, 0, 0]
    counts = [0, 0, 0]
    for _ in range(n):
        if random.random() < epsilon:
            ad = random.randint(0, 2)
        else:
            rates = [
                clicks[i] / counts[i] if counts[i] > 0 else 0
                for i in range(3)
            ]
            ad = np.argmax(rates)
        reward = int(random.random() < true_ctr[ad])
        clicks[ad] += reward
        counts[ad] += 1
    return sum(clicks) / n
def run_ucb():
    clicks = [0, 0, 0]
    counts = [0, 0, 0]
    for ad in range(3):
        reward = int(random.random() < true_ctr[ad])
        clicks[ad] += reward
        counts[ad] += 1
    for t in range(3, n + 1):
        ucb = [
            clicks[i] / counts[i]
            + math.sqrt(2 * math.log(t) / counts[i])
            for i in range(3)
        ]
        ad = np.argmax(ucb)
        reward = int(random.random() < true_ctr[ad])
        clicks[ad] += reward
        counts[ad] += 1
    return sum(clicks) / n
def run_thompson():
    clicks = [0, 0, 0]
    failures = [0, 0, 0]
    for _ in range(n):
        samples = [
            np.random.beta(
                clicks[i] + 1,
                failures[i] + 1
            )
            for i in range(3)
        ]
        ad = np.argmax(samples)
        reward = int(random.random() < true_ctr[ad])
        clicks[ad] += reward
        failures[ad] += 1 - reward
    return sum(clicks) / n
eg = run_epsilon_greedy()
ucb = run_ucb()
ts = run_thompson()
print("Click-Through Rates")
print("Epsilon-Greedy :", round(eg, 3))
print("UCB            :", round(ucb, 3))
print("Thompson       :", round(ts, 3))
results = {
    "Epsilon-Greedy": eg,
    "UCB": ucb,
    "Thompson Sampling": ts
}
print("\nBest Algorithm:",
      max(results, key=results.get))