import numpy as np
import random
tasks = ["Pick Box", "Deliver Box"]
Q1 = np.zeros((2, 2))
Q2 = np.zeros((2, 2))
alpha = 0.1
gamma = 0.9
for episode in range(100):
    state = random.randint(0, 1)
    action1 = np.argmax(Q1[state])
    action2 = np.argmax(Q2[state])
    reward1 = random.randint(5, 10)
    reward2 = random.randint(5, 10)
    Q1[state][action1] += alpha * (
        reward1 + gamma * np.max(Q1[state]) - Q1[state][action1]
    )
    Q2[state][action2] += alpha * (
        reward2 + gamma * np.max(Q2[state]) - Q2[state][action2]
    )
print("Robot 1 Q-Table")
print(np.round(Q1, 2))
print("\nRobot 2 Q-Table")
print(np.round(Q2, 2))