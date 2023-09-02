import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayMemory
from network import DeepQNetwork


class DQNAgent:
    def __init__(
        self,
        learning_rate,
        discount_factor,
        initial_epsilon,
        final_epsilon,
        epsilon_decay,
        obs_shape,
        n_actions,
        memory_size,
        batch_size,
        target_replace=1000,
        checkpoint_dir="checkpoints",
    ):
        # Learning params
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.action_space = [i for i in range(n_actions)]

        # Checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.checkpoint_path = checkpoint_dir

        # Networks
        self.q_network = DeepQNetwork(
            input_size=obs_shape[0],
            output_size=n_actions,
            checkpoint_path=self.checkpoint_path,
        )
        self.target_network = DeepQNetwork(
            input_size=obs_shape[0],
            output_size=n_actions,
            checkpoint_path=self.checkpoint_path,
        )
        self.optimizer = optim.Adam(
            params=self.q_network.parameters(),
            lr=self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.replace_frequency = target_replace
        self.replace_counter = 0

        # Replay memory
        self.replay_memory = ReplayMemory(
            obs_shape=obs_shape,
            buffer_size=memory_size,
        )
        self.batch_size = batch_size

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.get_greedy_action(obs)

    def get_greedy_action(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        with torch.no_grad():
            action_values = self.q_network(obs)
        action = torch.argmax(action_values).item()
        return action

    def store_transition(self, obs, action, obs_next, reward, terminal):
        self.replay_memory.store(obs, action, obs_next, reward, terminal)

    def replace_target_network(self):
        if self.replace_frequency % self.replace_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def serialise(self):
        self.q_network.serialise("q_network")
        self.target_network.serialise("target_network")

    def deserialise(self):
        self.q_network.deserialise(f"{self.checkpoint_path}/q_network.pt")
        self.target_network.deserialise(f"{self.checkpoint_path}/target_network.pt")

    def update(self):
        # Fill buffer before training
        if self.replay_memory.buffer_counter < self.batch_size:
            return

        # Sample transitions from memory
        obs, actions, obs_next, rewards, terminals = self.replay_memory.sample(
            self.batch_size
        )
        obs = torch.tensor(obs)
        actions = torch.tensor(actions)
        obs_next = torch.tensor(obs_next)
        rewards = torch.tensor(rewards)
        terminals = torch.tensor(terminals)

        self.optimizer.zero_grad()
        q_pred = self.q_network(obs)[torch.arange(0, self.batch_size), actions]
        q_next_pred = self.target_network(obs_next).max(dim=1)[0]
        q_next_pred[terminals] = 0
        q_target = rewards + self.discount_factor * q_next_pred

        loss = self.loss(q_target, q_pred)
        loss.backward()
        self.optimizer.step()

        self.replace_counter += 1
        self.replace_target_network()
        self.decay_epsilon()
