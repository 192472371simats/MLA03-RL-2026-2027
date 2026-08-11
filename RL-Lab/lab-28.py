import numpy as np
# 3 x 3 Grid
rows = 3
cols = 3
start = (0, 0)
goal = (2, 2)
gamma = 0.9
V = np.zeros((rows, cols))
actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]
for iteration in range(20):
    new_V = V.copy()
    for r in range(rows):
        for c in range(cols):
            if (r, c) == goal:
                continue
            values = []
            for dr, dc in actions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) == goal:
                        reward = 10
                    else:
                        reward = -1
                    value = reward + gamma * V[nr, nc]
                    values.append(value)
            new_V[r, c] = max(values)
    V = new_V
print("Optimal Value Function:")
print(np.round(V, 2))
state = start
path = [state]
visited = set()
visited.add(state)
while state != goal:
    r, c = state
    best_value = -float("inf")
    best_state = None
    for dr, dc in actions:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            next_state = (nr, nc)
            if next_state in visited:
                continue
            if V[nr, nc] > best_value:
                best_value = V[nr, nc]
                best_state = next_state
    if best_state is None:
        print("Path not found")
        break
    state = best_state
    visited.add(state)
    path.append(state)
print("\nOptimal Path:")
print(path)