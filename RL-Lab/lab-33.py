import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
actor = keras.Sequential([
    keras.Input(shape=(3,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(2, activation="sigmoid")
])
state_input = keras.Input(shape=(3,))
action_input = keras.Input(shape=(2,))
x = layers.Concatenate()(
    [state_input, action_input]
)
x = layers.Dense(
    32,
    activation="relu"
)(x)
x = layers.Dense(
    32,
    activation="relu"
)(x)
critic_output = layers.Dense(1)(x)
critic = keras.Model(
    [state_input, action_input],
    critic_output
)
actor_optimizer = keras.optimizers.Adam(
    learning_rate=0.001
)
critic_optimizer = keras.optimizers.Adam(
    learning_rate=0.001
)
state = tf.constant(
    [[0.7, 0.3, 0.5]],
    dtype=tf.float32
)
with tf.GradientTape() as actor_tape:
    action = actor(state)
    q_value = critic(
        [state, action]
    )
    actor_loss = -tf.reduce_mean(q_value)
actor_gradients = actor_tape.gradient(
    actor_loss,
    actor.trainable_variables
)
actor_optimizer.apply_gradients(
    zip(
        actor_gradients,
        actor.trainable_variables
    )
)
target_reward = tf.constant(
    [[1.0]],
    dtype=tf.float32
)
with tf.GradientTape() as critic_tape:
    predicted_q = critic(
        [state, action]
    )
    critic_loss = tf.reduce_mean(
        tf.square(
            target_reward - predicted_q
        )
    )
critic_gradients = critic_tape.gradient(
    critic_loss,
    critic.trainable_variables
)
critic_optimizer.apply_gradients(
    zip(
        critic_gradients,
        critic.trainable_variables
    )
)
print("DDPG Training Step Completed")
print("\nInput State:")
print(state.numpy()[0])
print("\nActor Output:")
print(
    np.round(
        action.numpy()[0],
        3
    )
)
print("\nCritic Q-Value:")
print(
    np.round(
        q_value.numpy()[0][0],
        3
    )
)
print("\nActor Loss:")
print(
    round(
        float(actor_loss.numpy()),
        3
    )
)
print("\nCritic Loss:")
print(
    round(
        float(critic_loss.numpy()),
        3
    )
)