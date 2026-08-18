import random
Q = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]
alpha = 0.5
gamma = 0.9
epsilon = 0.2
for episode in range(200):
    for state in range(3):
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = Q[state].index(
                max(Q[state])
            )
        if state == 0:
            if action == 0:
                reward = 8
            else:
                reward = -3
        elif state == 1:
            if action == 1:
                reward = 8
            else:
                reward = 2
        else:
            if action == 2:
                reward = 8
            else:
                reward = 3
        Q[state][action] += alpha * (
            reward
            + gamma * max(Q[state])
            - Q[state][action]
        )
print("Personalized Learning Training Completed")
states = [
    "Weak Student",
    "Average Student",
    "Strong Student"
]
lessons = [
    "Easy Lesson",
    "Medium Lesson",
    "Difficult Lesson"
]
print("\nPersonalized Policy:")
for state in range(3):
    best_action = Q[state].index(
        max(Q[state])
    )
    print(
        states[state],
        "->",
        lessons[best_action]
    )
print("\nQ Table:")
for row in Q:
    print(
        [round(x, 2) for x in row]
    )