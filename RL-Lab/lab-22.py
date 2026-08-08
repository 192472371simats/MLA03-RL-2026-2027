import numpy as np
import random
size = 4
Q = np.zeros((size, size, 4))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
food = (3, 3)
ghost = (1, 1)
def move(state, action):
    x, y = state
    if action == 0:
        x = max(0, x - 1)
    elif action == 1:
        x = min(size - 1, x + 1)
    elif action == 2:
        y = max(0, y - 1)
    else:
        y = min(size - 1, y + 1)
    return (x, y)
for episode in range(1000):
    state = (0, 0)
    for step in range(50):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        next_state = move(state, action)
        if next_state == food:
            reward = 10
            done = True
        elif next_state == ghost:
            reward = -10
            done = True
        else:
            reward = -1
            done = False
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state])
            - Q[state][action]
        )
        state = next_state
        if done:
            break
print("Training Completed")
state = (0, 0)
print("\nLearned Path:")
for _ in range(15):
    print(state)
    if state == food:
        break
    action = np.argmax(Q[state])
    state = move(state, action)