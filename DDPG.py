import gym
from gym import wrappers
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
# get_ipython().magic('matplotlib inline')

from ddpg_agent import Agent


# ### 2. Instantiate the Environment and Agent

from gym.envs import box2d
env = gym.make("BipedalWalker-v2")
env.seed(8)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=8)

VIDEO_SUBDIR = "./vid/"
timestamp = str(time())

def make_video_frames(i_episode):
    global env
    env = wrappers.Monitor(env, VIDEO_SUBDIR + timestamp + "/" + str(i_episode) + "/")
    state = env.reset()
    agent.reset()
    while True:
        action = agent.act(state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
      
    env.close()

# ### 3. Train the Agent with DDPG
n_episodes = 301
latest_actor_fname = ""
latest_critic_fname = ""

def ddpg(max_t=500):
    scores_deque = deque(maxlen=100)
    scores = []
    max_reward = -np.Inf
    durations = deque(maxlen=100)
    for i_episode in range(n_episodes):
        state = env.reset()
        agent.reset()
        score = 0    
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                durations.append(t)
                break
        if (not done):
            durations.append(max_t)
            
        if (score > max_reward):
            max_reward = score
            
        scores_deque.append(score)
        scores.append(score)
        outstr = '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tav duration: {:.1f}'
        print(outstr.format(i_episode, np.mean(scores_deque), score, np.mean(durations)), end="")
        if i_episode % 50 == 0 or i_episode == n_episodes - 1:
            latest_actor_fname = 'checkpoint_actor' + str(i_episode) + '.pth'
            latest_critic_fname = 'checkpoint_critic' + str(i_episode) + '.pth'
            torch.save(agent.actor_local.state_dict(), latest_actor_fname)
            torch.save(agent.critic_local.state_dict(), latest_critic_fname)
            print('\rEpisode {}\tAverage Score: {:.2f}\tmax_reward: {:.1f}    '.format(i_episode, np.mean(scores_deque), max_reward))
            make_video_frames(i_episode)
            
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



agent.actor_local.load_state_dict(torch.load(latest_actor_fname))
agent.critic_local.load_state_dict(torch.load(latest_critic_fname))

state = env.reset()
agent.reset()   
while True:
    action = agent.act(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break
        
env.close()
