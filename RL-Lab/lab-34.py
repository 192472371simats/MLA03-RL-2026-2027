import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
model = tf.keras.Sequential([
    layers.Dense(
        16,
        activation="relu",
        input_shape=(1,)
    ),
    layers.Dense(3, activation="softmax")
])
optimizer = tf.keras.optimizers.Adam(0.01)
actions = [
    "Heat",
    "Cool",
    "No Change"
]
temperatures = [18, 20, 23, 27, 30]
gamma = 0.9
with tf.GradientTape() as tape:
    log_probs = []
    rewards = []
    for temperature in temperatures:
        state = np.array(
            [[temperature]],
            dtype=np.float32
        )
        probabilities = model(state)[0]
        action = np.random.choice(
            3,
            p=probabilities.numpy()
        )
        if 21 <= temperature <= 24:
            comfort_reward = 5
        else:
            comfort_reward = -5
        if action == 2:
            energy_cost = 0
        else:
            energy_cost = -2
        reward = comfort_reward + energy_cost
        log_prob = tf.math.log(
            probabilities[action] + 1e-8
        )
        log_probs.append(log_prob)
        rewards.append(reward)
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = np.array(
        returns,
        dtype=np.float32
    )
    if np.std(returns) > 0:
        returns = (
            returns - np.mean(returns)
        ) / (
            np.std(returns) + 1e-8
        )
    loss = 0
    for log_prob, G in zip(
        log_probs,
        returns
    ):
        loss -= log_prob * G
gradients = tape.gradient(
    loss,
    model.trainable_variables
)
optimizer.apply_gradients(
    zip(
        gradients,
        model.trainable_variables
    )
)
print("Temperature Control Results")
print("----------------------------")
for i, temperature in enumerate(
    temperatures
):
    print(
        "Temperature:",
        temperature,
        "Reward:",
        rewards[i]
    )
print("\nREINFORCE Training Completed")