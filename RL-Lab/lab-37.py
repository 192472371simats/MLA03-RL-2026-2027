import random
states = ["Left", "Center", "Right"]
actions = ["Left", "Right"]
belief = {
    "Left": 0.33,
    "Center": 0.34,
    "Right": 0.33
}
position = 0
goal = 2
for step in range(10):
    if belief["Right"] > belief["Left"]:
        action = "Right"
    else:
        action = "Left"
    if action == "Right":
        position = min(position + 1, 2)
    else:
        position = max(position - 1, 0)
    if random.random() < 0.8:
        observation = states[position]
    else:
        observation = random.choice(states)
    for state in states:
        if state == observation:
            belief[state] = 0.8
        else:
            belief[state] = 0.1
    print(
        "Step:", step + 1,
        "Action:", action,
        "Observation:", observation
    )
    if position == goal:
        print("Goal Reached!")
        break
print("\nFinal Belief:")
print(belief)