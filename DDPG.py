import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

from ddpg_agent import Agent


# ### 2. Instantiate the Environment and Agent

from gym.envs import box2d
env = gym.make("BipedalWalker-v2")
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)


# ### 3. Train the Agent with DDPG
# 
# Run the code cell below to train the agent from scratch.  Alternatively, you can skip to the next code cell to load the pre-trained weights from file.
def ddpg(n_episodes=4000, max_t=500):
    scores_deque = deque(maxlen=100)
    scores = []
    max_reward = -np.Inf
    durations = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
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
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor' + str(i_episode) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic' + str(i_episode) + '.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\tmax_reward: {:.1f}    '.format(i_episode, np.mean(scores_deque), max_reward))
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### 4. Watch a Smart Agent!
# 
# In the next code cell, you will load the trained weights from file to watch a smart agent!


agent.actor_local.load_state_dict(torch.load('checkpoint_actor4000.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic4000.pth'))

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


# ### 5. Explore
# 
# In this exercise, we have provided a sample DDPG agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster than this benchmark implementation.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task!
# - Write your own DDPG implementation.  Use this code as reference only when needed -- try as much as you can to write your own algorithm from scratch.
# - You may also like to implement prioritized experience replay, to see if it speeds learning.  
# - The current implementation adds Ornsetein-Uhlenbeck noise to the action space.  However, it has [been shown](https://blog.openai.com/better-exploration-with-parameter-noise/) that adding noise to the parameters of the neural network policy can improve performance.  Make this change to the code, to verify it for yourself!
# - Write a blog post explaining the intuition behind the DDPG algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
