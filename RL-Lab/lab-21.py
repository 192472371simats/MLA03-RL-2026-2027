import numpy as np
import random
Q = np.zeros((3, 2))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
for episode in range(500):
    state = random.randint(0, 2)
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        action = np.argmax(Q[state])
    # Reward
    if action == 1:              # Save energy
        reward = 10 if state == 2 else 5
    else:                        # Normal operation
        reward = 3 if state == 0 else -5
    next_state = max(0, state - 1) if action == 1 else min(2, state + 1)
    Q[state, action] += alpha * (
        reward + gamma * np.max(Q[next_state])
        - Q[state, action]
    )
print("Learned Q-Table:")
print(np.round(Q, 2))
print("\nBest Actions:")
for state in range(3):
    print("State", state, "-> Action", np.argmax(Q[state]))