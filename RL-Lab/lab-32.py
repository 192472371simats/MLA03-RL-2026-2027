import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
inputs = keras.Input(shape=(4,))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
value = layers.Dense(1)(x)
advantage = layers.Dense(4)(x)
mean_advantage = ops.mean(
    advantage,
    axis=1,
    keepdims=True
)
q_values = value + advantage - mean_advantage
model = keras.Model(
    inputs=inputs,
    outputs=q_values
)
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)
goal = np.array([3, 3])
actions = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1)     # Right
]
def move(state, action):
    new_state = state + np.array(
        actions[action]
    )
    new_state = np.clip(
        new_state,
        0,
        3
    )
    return new_state
memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.98
for episode in range(100):
    state = np.array(
        [0, 0],
        dtype=np.float32
    )
    for step in range(30):
        state_input = np.array([
            state[0],
            state[1],
            goal[0],
            goal[1]
        ], dtype=np.float32)
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            q = model.predict(
                state_input.reshape(1, -1),
                verbose=0
            )
            action = np.argmax(q[0])
        next_state = move(
            state,
            action
        )
        if np.array_equal(
            next_state,
            goal
        ):
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        next_input = np.array([
            next_state[0],
            next_state[1],
            goal[0],
            goal[1]
        ], dtype=np.float32)
        memory.append(
            (
                state_input,
                action,
                reward,
                next_input,
                done
            )
        )
        state = next_state
        if len(memory) >= 32:
            batch = random.sample(
                memory,
                32
            )
            states = np.array(
                [x[0] for x in batch],
                dtype=np.float32
            )
            actions_batch = np.array(
                [x[1] for x in batch]
            )
            rewards = np.array(
                [x[2] for x in batch],
                dtype=np.float32
            )
            next_states = np.array(
                [x[3] for x in batch],
                dtype=np.float32
            )
            dones = np.array(
                [x[4] for x in batch]
            )
            current_q = model.predict(
                states,
                verbose=0
            )
            next_q = model.predict(
                next_states,
                verbose=0
            )
            for i in range(32):
                target = rewards[i]
                if not dones[i]:
                    target += (
                        gamma *
                        np.max(next_q[i])
                    )
                current_q[
                    i,
                    actions_batch[i]
                ] = target
            model.fit(
                states,
                current_q,
                epochs=1,
                verbose=0
            )
        if done:
            break
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )
    if episode % 20 == 0:
        print(
            "Episode:",
            episode,
            "Epsilon:",
            round(epsilon, 3)
        )
print("\nDueling DQN Training Completed")
state = np.array(
    [0, 0],
    dtype=np.float32
)
state_input = np.array([
    state[0],
    state[1],
    goal[0],
    goal[1]
], dtype=np.float32)
q = model.predict(
    state_input.reshape(1, -1),
    verbose=0
)
print("\nStarting State:", state)
print(
    "Q-Values:",
    np.round(q[0], 2)
)
action_names = [
    "Up",
    "Down",
    "Left",
    "Right"
]
best_action = np.argmax(q[0])
print(
    "Best Action:",
    action_names[best_action]
)