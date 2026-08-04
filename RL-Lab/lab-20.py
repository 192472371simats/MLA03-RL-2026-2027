import numpy as np
states = ["Safe", "Victim Found"]
belief = np.array([0.7, 0.3])
actions = ["Search", "Rescue"]
print("Initial Belief")
for i in range(len(states)):
    print(states[i], ":", belief[i])
belief = np.array([0.3, 0.7])
print("\nUpdated Belief")
for i in range(len(states)):
    print(states[i], ":", belief[i])
action = actions[np.argmax(belief)]
print("\nSelected Action:", action)