import numpy as np
import tensorflow as tf
import random
import copy
from collections import namedtuple, deque
from Model import Actor, Critic

#TODO: Check OUNoise for mistakes, adjust Replay_Buffer terminology, add actual Off-Policy Agent

class OUNoise(object):
    "Ornstein-Uhlenbeck process"
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        "Initialize parameters and noise process."
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        "Reset the internal state (= noise) to mean (mu)."
        self.state = copy.copy(self.mu)

    def sample(self):
        "Update internal state and return it as a noise sample."
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size = self.mu.size)
        return self.state

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return (states, actions, rewards, next_states, terminal)
