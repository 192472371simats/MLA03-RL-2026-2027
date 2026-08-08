import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
model = tf.keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(3,)),
    layers.Dense(3, activation="softmax")
])
optimizer = tf.keras.optimizers.Adam(0.01)
actions = ["Buy", "Sell", "Hold"]
states = np.array([
    [0.05,  1, 0.02],
    [0.03,  1, 0.03],
    [-0.04, -1, 0.05],
    [0.02,  1, 0.02]
], dtype=np.float32)
gamma = 0.9
with tf.GradientTape() as tape:
    log_probs = []
    rewards = []
    for state in states:
        state_input = state.reshape(1, 3)
        probabilities = model(state_input)[0]
        action = np.random.choice(
            3,
            p=probabilities.numpy()
        )
        price_change = float(state[0])
        if action == 0:          # Buy
            reward = price_change
        elif action == 1:        # Sell
            reward = -price_change
        else:                    # Hold
            reward = 0.0
        log_prob = tf.math.log(
            probabilities[action] + 1e-8
        )
        log_probs.append(log_prob)
        rewards.append(reward)
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = np.array(returns, dtype=np.float32)
    if np.std(returns) > 0:
        returns = (
            returns - np.mean(returns)
        ) / (np.std(returns) + 1e-8)
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G
gradients = tape.gradient(
    loss,
    model.trainable_variables
)
optimizer.apply_gradients(
    zip(gradients, model.trainable_variables)
)
print("Trading Results")
print("----------------")
for i, reward in enumerate(rewards):
    print(
        "Step", i + 1,
        "Reward:", round(reward, 3)
    )
print("\nTotal Profit:",
      round(sum(rewards), 3))
print("REINFORCE training completed.")