import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(32, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(3, activation="linear")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)
target_model = Sequential([
    Dense(32, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(3, activation="linear")
])
target_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)
target_model.set_weights(model.get_weights())
memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.98
actions = [
    "Keep Lane",
    "Change Left",
    "Change Right"
]
for episode in range(50):
    state = np.array(
        [1, 60, 30, 0.2],
        dtype=np.float32
    )
    for step in range(20):
        if random.random() < epsilon:
            action = random.randrange(3)
        else:
            q_values = model.predict(
                state.reshape(1, -1),
                verbose=0
            )
            action = np.argmax(q_values[0])
        if action == 0:
            reward = 5
        elif action == 1:
            reward = 3
        else:
            reward = 3
        next_state = state.copy()
        next_state[1] += random.uniform(-2, 2)
        done = (step == 19)
        memory.append(
            (
                state,
                action,
                reward,
                next_state,
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
            next_q = target_model.predict(
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
    if episode % 5 == 0:
        target_model.set_weights(
            model.get_weights()
        )
    if episode % 10 == 0:
        print(
            "Episode:",
            episode,
            "Epsilon:",
            round(epsilon, 3)
        )
print("\nDQN Training Completed")
test_state = np.array(
    [[1, 60, 30, 0.2]],
    dtype=np.float32
)
q_values = model.predict(
    test_state,
    verbose=0
)[0]
print("\nQ-Values:")
print(np.round(q_values, 2))
best_action = np.argmax(q_values)
print(
    "Selected Action:",
    actions[best_action]
)