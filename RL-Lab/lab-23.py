import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# Policy network
model = tf.keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(4,)),
    layers.Dense(3, activation="softmax")
])
optimizer = tf.keras.optimizers.Adam(0.001)
state = np.array([[1, 60, 30, 0.4]], dtype=np.float32)
old_prob = model(state).numpy()[0]
action = np.random.choice(3, p=old_prob)
actions = ["Keep Lane", "Change Left", "Change Right"]
reward = 10 if action == 0 else 8
with tf.GradientTape() as tape:
    prob = model(state)[0, action]
    ratio = prob / (old_prob[action] + 1e-8)
    epsilon = 0.2
    clipped = tf.clip_by_value(
        ratio,
        1 - epsilon,
        1 + epsilon
    )
    loss = -tf.minimum(
        ratio * reward,
        clipped * reward
    )
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(
    zip(gradients, model.trainable_variables)
)
print("Vehicle State:", state[0])
print("Selected Action:", actions[action])
print("Reward:", reward)