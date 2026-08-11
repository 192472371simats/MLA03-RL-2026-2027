import numpy as np
states = 3
actions = 2
gamma = 0.9
reward = np.array([
    [5, 3],
    [3, 6],
    [1, 10]
])
# Initial policy
policy = np.zeros(states, dtype=int)
V = np.zeros(states)
for iteration in range(20):
    # Policy Evaluation
    for _ in range(50):
        new_V = np.zeros(states)
        for s in range(states):
            a = policy[s]
            # Simplified transition:
            # traffic tends to move to the next state
            next_state = min(s + 1, states - 1)
            new_V[s] = (
                reward[s, a]
                + gamma * V[next_state]
            )
        V = new_V
    # Policy Improvement
    stable = True
    for s in range(states):
        old_action = policy[s]
        values = []
        for a in range(actions):
            next_state = min(s + 1, states - 1)
            values.append(
                reward[s, a]
                + gamma * V[next_state]
            )
        policy[s] = np.argmax(values)
        if old_action != policy[s]:
            stable = False
    if stable:
        break
print("Optimal State Values:")
print(np.round(V, 2))
print("\nOptimal Traffic Light Policy:")
names = [
    "North-South Green",
    "East-West Green"
]
for s in range(states):
    print(
        "State", s,
        "->", names[policy[s]]
    )