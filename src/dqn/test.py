import gymnasium as gym
import numpy as np
from agent import DQNAgent
from tqdm import tqdm

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = DQNAgent(
        learning_rate=1e-4,
        discount_factor=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay=1e-6,
        obs_shape=(8,),
        n_actions=4,
        memory_size=10000,
        batch_size=5,
    )
    agent.deserialise()

    for _ in range(100):
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.get_greedy_action(obs)
            obs_next, reward, done, _, _ = env.step(action)
            env.render()
            obs = obs_next
