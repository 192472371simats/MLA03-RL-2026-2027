import numpy as np
import random
import math
prices = [80, 100, 120]
# Probability that a customer buys at each price
buy_probability = [0.75, 0.60, 0.40]
rounds = 1000
epsilon = 0.1
def epsilon_greedy():
    revenue = [0, 0, 0]
    count = [0, 0, 0]
    for _ in range(rounds):
        if random.random() < epsilon:
            arm = random.randint(0, 2)
        else:
            avg = [
                revenue[i] / count[i] if count[i] > 0 else 0
                for i in range(3)
            ]
            arm = np.argmax(avg)
        sale = random.random() < buy_probability[arm]
        if sale:
            reward = prices[arm]
        else:
            reward = 0
        revenue[arm] += reward
        count[arm] += 1
    return sum(revenue)
def ucb():
    revenue = [0, 0, 0]
    count = [0, 0, 0]
    for arm in range(3):
        sale = random.random() < buy_probability[arm]
        if sale:
            revenue[arm] += prices[arm]
        count[arm] += 1
    for t in range(4, rounds + 1):
        ucb_values = []
        for i in range(3):
            avg = revenue[i] / count[i]
            confidence = math.sqrt(
                2 * math.log(t) / count[i]
            )
            ucb_values.append(avg + confidence)
        arm = np.argmax(ucb_values)
        sale = random.random() < buy_probability[arm]
        if sale:
            revenue[arm] += prices[arm]
        count[arm] += 1
    return sum(revenue)
def thompson_sampling():
    successes = [0, 0, 0]
    failures = [0, 0, 0]
    revenue = 0
    for _ in range(rounds):
        samples = [
            np.random.beta(
                successes[i] + 1,
                failures[i] + 1
            )
            for i in range(3)
        ]
        arm = np.argmax(samples)
        sale = random.random() < buy_probability[arm]
        if sale:
            successes[arm] += 1
            revenue += prices[arm]
        else:
            failures[arm] += 1
    return revenue
eg = epsilon_greedy()
ucb_result = ucb()
ts = thompson_sampling()
print("Total Revenue")
print("----------------------")
print("Epsilon-Greedy :", eg)
print("UCB            :", ucb_result)
print("Thompson       :", ts)
results = {
    "Epsilon-Greedy": eg,
    "UCB": ucb_result,
    "Thompson Sampling": ts
}
print("\nBest Strategy:",
      max(results, key=results.get))