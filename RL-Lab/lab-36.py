import random
Q = {
    "Collect": {"move": 0.0, "pickup": 0.0},
    "Deliver": {"move": 0.0, "drop": 0.0}
}
alpha = 0.5
gamma = 0.9
epsilon = 0.2
for episode in range(100):
    resource = False
    total_reward = 0
    # ---------- Collect ----------
    action = random.choice(
        ["move", "pickup"]
    ) if random.random() < epsilon else max(
        Q["Collect"],
        key=Q["Collect"].get
    )
    if action == "pickup":
        resource = True
        reward = 5
    else:
        reward = -1
    Q["Collect"][action] += alpha * (
        reward
        - Q["Collect"][action]
    )
    total_reward += reward
    # ---------- Deliver ----------
    if resource:
        action = random.choice(
            ["move", "drop"]
        ) if random.random() < epsilon else max(
            Q["Deliver"],
            key=Q["Deliver"].get
        )
        if action == "drop":
            reward = 10
        else:
            reward = -1
        Q["Deliver"][action] += alpha * (
            reward
            - Q["Deliver"][action]
        )
        total_reward += reward
print("MAXQ Training Completed")
print("\nLearned Collect Policy:")
print(max(Q["Collect"], key=Q["Collect"].get))
print("\nLearned Deliver Policy:")
print(max(Q["Deliver"], key=Q["Deliver"].get))
print("\nQ Values:")
print(Q)