import numpy as np


class ReplayMemory:
    def __init__(self, obs_shape, buffer_size):
        # Setup buffers
        self.obs_memory = np.zeros((buffer_size, *obs_shape), np.float32)
        self.action_memory = np.zeros((buffer_size,), np.int32)
        self.obs_next_memory = np.zeros((buffer_size, *obs_shape), np.float32)
        self.reward_memory = np.zeros((buffer_size,), np.float32)
        self.terminal_memory = np.zeros((buffer_size,), np.bool_)

        # Setup index counter
        self.buffer_size = buffer_size
        self.buffer_counter = 0

    def store(self, obs, action, obs_next, reward, terminal):
        # Buffer index where transition inserted
        idx = self.buffer_counter % self.buffer_size
        self.buffer_counter += 1

        # Store transition in buffer
        self.obs_memory[idx] = obs
        self.action_memory[idx] = action
        self.obs_next_memory[idx] = obs_next
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = terminal

    def sample(self, batch_size):
        # Get index of transitions in batch
        self.max_idx = min(self.buffer_counter, self.buffer_size)
        batch_idx = np.random.choice(self.max_idx, size=batch_size, replace=False)

        # Extract transitions from buffer
        obs = self.obs_memory[batch_idx]
        actions = self.action_memory[batch_idx]
        obs_next = self.obs_next_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        terminals = self.terminal_memory[batch_idx]
        return obs, actions, obs_next, rewards, terminals
