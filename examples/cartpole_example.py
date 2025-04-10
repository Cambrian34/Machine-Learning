import gym

# Create environment with render mode
env = gym.make("CartPole-v1", render_mode="human")

# Reset returns (observation, info)
observation, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Choose random action
    # Step returns 5 values now
    observation, reward, terminated, truncated, info = env.step(action)

    # Episode is done when either terminated or truncated is True
    if terminated or truncated:
        observation, _ = env.reset()

env.close()