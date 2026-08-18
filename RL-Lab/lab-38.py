import random
Q = [[0.0, 0.0] for _ in range(3)]
alpha = 0.5
gamma = 0.9
epsilon = 0.2
for episode in range(100):
    for state in range(3):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = Q[state].index(
                max(Q[state])
            )
        if state == 2 and action == 1:
            reward = 10
        elif state == 2 and action == 0:
            reward = -10
        elif action == 1:
            reward = 4
        else:
            reward = 2
        Q[state][action] += alpha * (
            reward
            + gamma * max(Q[state])
            - Q[state][action]
        )
print("Healthcare RL Training Completed")
names = [
    "Low Urgency",
    "Medium Urgency",
    "High Urgency"
]
actions = [
    "Normal Treatment",
    "Priority Treatment"
]
print("\nLearned Policy:")
for state in range(3):
    best = Q[state].index(
        max(Q[state])
    )
    print(
        names[state],
        "->",
        actions[best]
    )
print("\nQ Table:")
for row in Q:
    print(
        [round(x, 2) for x in row]
    )