import gym

env = gym.make("FrozenLake-v1", render_mode=None)

state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()   # Random action
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Episode Reward:", total_reward)

env.close()