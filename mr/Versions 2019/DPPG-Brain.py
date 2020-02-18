#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports here
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import logging

import imageio
import io
import gin
import tensorflow as tf
import gym
from gym import wrappers

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from gym.envs import box2d
import argparse


# In[2]:


#Defaults
env_name = 'BipedalWalker-v2'
num_iterations = 3000000
# Params for collect
initial_collect_steps = 1000 #1000
collect_steps_per_iteration = 1
num_parallel_environments = 1
replay_buffer_capacity = 100000
ou_stddev = 0.2 #0.2
ou_damping = 0.15 #0.15
# Params for target update
target_update_tau = 0.05
target_update_period = 5
# Params for train
train_steps_per_iteration = 1 #1
batch_size = 64
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
dqda_clipping = None
td_errors_loss_fn = tf.compat.v1.losses.mean_squared_error #tf.compat.v1.losses.huber_loss potential problem? MSE?
gamma = 0.99 #0.995
reward_scale_factor = 1.0
gradient_clipping = None
use_tf_functions = True
# Params for eval
num_eval_episodes = 10
eval_interval = 10000
eval_metrics_callback = None
log_interval = 1000
summary_interval = 1000
summaries_flush_secs = 10
run_id = 210120
root_dir = '~/'

#For Brain
use_brain = True


# In[3]:


if use_brain:
    global args
    parser = argparse.ArgumentParser(description = 'DDPG Arguments')
    parser.add_argument('--run_id', type = int, help = "identifying substring for folder names (default: date)")
    parser.add_argument('--root_dir', help = "directory for output ")
    args = parser.parse_args()
    
    if args.run_id is not None:
        run_id = args.run_id
    if root_dir is not None
        root_dir = args.root_dir


# In[4]:


def DDPG_Bipedal(root_dir):
    
    #Setting up directories for log and evaluation
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train' + str(run_id))
    eval_dir = os.path.join(root_dir, 'eval' + str(run_id))
    video_dir = os.path.join(root_dir, 'vid' + str(run_id))
    
    #Set up train summary writer and eval summary writer
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis = summaries_flush_secs * 1000
    )
    train_summary_writer.set_as_default()
    
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis = summaries_flush_secs * 1000
    )
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size = num_eval_episodes), #metric to compute av return
        tf_metrics.AverageEpisodeLengthMetric(buffer_size = num_eval_episodes) #metric to compute av ep length
    ]
    
    #Create global step
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
        eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
        eval_py_env = suite_gym.load(env_name)
    
    
        #Define Actor Network
        actorNN = actor_network.ActorNetwork(
                  tf_env.time_step_spec().observation,
                  tf_env.action_spec(),
                  fc_layer_params=(400, 300),
        )
    
        #Define Critic Network
        NN_input_specs = (tf_env.time_step_spec().observation,
                          tf_env.action_spec()
        )
    
        criticNN = critic_network.CriticNetwork(
                   NN_input_specs,
                   observation_fc_layer_params = (400,),
                   action_fc_layer_params = None,
                   joint_fc_layer_params = (300,),
        )
        
        #Define & initialize DDPG Agent
        agent = ddpg_agent.DdpgAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                actor_network = actorNN,
                critic_network = criticNN,
                actor_optimizer = tf.compat.v1.train.AdamOptimizer(
                                  learning_rate = actor_learning_rate),
                critic_optimizer = tf.compat.v1.train.AdamOptimizer(
                                   learning_rate = critic_learning_rate),
                ou_stddev = ou_stddev,
                ou_damping = ou_damping,
                target_update_tau = target_update_tau,
                target_update_period = target_update_period,
                dqda_clipping = None,
                td_errors_loss_fn = tf.compat.v1.losses.mean_squared_error,
                gamma = gamma,
                reward_scale_factor = 1.0,
                gradient_clipping = None,
                debug_summaries = False,
                summarize_grads_and_vars = False,
                train_step_counter = global_step
        )
        agent.initialize()
        
        #Determine which train metrics to display with summary writer
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        #Set policies for evaluation and initial collection
        eval_policy = agent.policy #actor policy
        collect_policy = agent.collect_policy #actor policy with OUNoise
        
        #Set up replay buffer
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        agent.collect_data_spec,
                        batch_size = tf_env.batch_size,
                        max_length = replay_buffer_capacity
        )
        
        #Define driver for initial replay buffer filling
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                                     tf_env,
                                     collect_policy,
                                     observers = [replay_buffer.add_batch],
                                     num_steps = initial_collect_steps
        )

        #Define collect driver for collect steps per iteration
        collect_driver = dynamic_step_driver.DynamicStepDriver(
                             tf_env,
                             collect_policy,
                             observers = [replay_buffer.add_batch] + train_metrics,
                             num_steps = collect_steps_per_iteration
        )
        
        if use_tf_functions:
            initial_collect_driver.run = common.function(initial_collect_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            agent.train = common.function(agent.train)
            
        # Collect initial replay data
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps with '
            'a random policy.', initial_collect_steps)
        initial_collect_driver.run()
        
        #Computes Evaluation Metrics
        results = metric_utils.eager_compute(
                  eval_metrics,
                  eval_tf_env,
                  eval_policy,
                  num_episodes = num_eval_episodes,
                  train_step = global_step,
                  summary_writer = eval_summary_writer,
                  summary_prefix = 'Metrics',
        )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)
        
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0 

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
                  num_parallel_calls = 3,
                  sample_batch_size = 64,
                  num_steps = 2).prefetch(3)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator) #Get experience from dataset
            return agent.train(experience) #Train agent on that experience
        
        if use_tf_functions:
            train_step = common.function(train_step)
            
        
        #Where the magic happens
        for _ in range(num_iterations):
            start_time = time.time() #Get start time
            #Collect some data for replay buffer (also observed by train metrics)
            time_step, policy_state = collect_driver.run(
                                      time_step = time_step,
                                      policy_state = policy_state,
            )
            #Train on collected experience n times
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                             train_loss.loss
                )
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name = 'global_steps_per_sec', data = steps_per_sec, 
                    step = global_step
                )
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step = global_step, 
                                          step_metrics = train_metrics[:2])
                
            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(
                          eval_metrics,
                          eval_tf_env,
                          eval_policy,
                          num_episodes = num_eval_episodes,
                          train_step = global_step,
                          summary_writer = eval_summary_writer,
                          summary_prefix = 'Metrics',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(eval_metrics)
                if results['AverageReturn'].numpy() >= 270.0:
                    num_episodes = 5
                    frames = []
                    for _ in range(num_episodes):
                        time_step = eval_tf_env.reset()
                        frames.append(eval_py_env.render())
                        while not time_step.is_last():
                            action_step = eval_policy.action(time_step)
                            time_step = eval_tf_env.step(action_step.action)
                            fnally = action_step[0].numpy()
                            next_state, reward, done, _ = eval_py_env.step(fnally[0])
                            frames.append(eval_py_env.render())
                        eval_py_env.close()
                        eval_tf_env.close()
                    gif_file = root_dir + "/" + run_id + "-" + str(global_step.numpy()) + '.gif'
                    imageio.mimsave(gif_file, frames, format = 'gif', fps = 60)
            
    return train_loss


DDPG_Bipedal(root_dir)


# In[5]:


get_ipython().run_line_magic('tb', '')


# In[ ]:




