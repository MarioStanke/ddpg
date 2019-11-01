import numpy as np
import tensorflow as tf
import random
import copy
from collections import namedtuple, deque
from Model import Actor, Critic
from mr.TF_Implementations.keiohta.target_update_ops import update_target_variables

#TODO: Check OUNoise for mistakes, adjust Replay_Buffer terminology, add actual Off-Policy Agent
#NOTES: Removed seed from critic/actor init, does tf.keras optim use weight decay?

BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 64                      # minibatch size
REPLAY_START_SIZE = BATCH_SIZE       # start training when this many examples were collected
GAMMA = 0.99                         # discount factor
tau=0.005                      # for soft update of target parameters
LR_ACTOR = 0.001             # learning rate of the actor 
LR_CRITIC = 0.001           # learning rate of the critic
#WEIGHT_DECAY = 0.0001 # L2 weight decay
sigma=0.1


class Agent():
    def __init__(self, state_size, action_size, lows, highs, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lows = lows
        self.highs = highs
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,self.actor.weights, tau=1.)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        update_target_variables(self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)








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
