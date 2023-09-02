import gymnasium as gym
import numpy as np
from agent import DQNAgent
from tqdm import tqdm

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode=None)
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

    scores = []
    max_score = -10000
    pbar = tqdm(range(100000))

    for episode in pbar:
        rtn = 0
        done = False
        obs, _ = env.reset()

        for _ in range(1000):
            action = agent.get_action(obs)
            obs_next, reward, done, _, _ = env.step(action)

            rtn += reward
            agent.store_transition(obs, action, obs_next, reward, done)
            agent.update()

            obs = obs_next
            if done:
                break
        scores.append(rtn)

        if episode % 25 == 0:
            pbar.set_description(
                f"Scores: {np.mean(scores[-100:]):.3f} Eps: {agent.epsilon:.5f}"
            )
        if episode % 100 == 0 and episode > 0:
            if np.mean(scores[-100:]) > max_score:
                max_score = np.mean(scores[-100:])
                agent.serialise()
