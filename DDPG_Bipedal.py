from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

from absl import app
from absl import logging

import gin
import tensorflow as tf
import gym
from gym import wrappers

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from gym.envs import box2d

'''Hyperparameter Settings'''

# Defaults
env_name = 'BipedalWalker-v2'
num_iterations = 2500000
use_tf_functions = True

# Replay Buffer Parameters & Noise Function Parameters
initial_collect_steps = 1000 
collect_steps_per_iteration = 1
replay_buffer_capacity = 100000
ou_stddev = 0.2 
ou_damping = 0.15 

# Target Update Parameters
target_update_tau = 0.05
target_update_period = 5

# Train Step Parameters
train_steps_per_iteration = 1 
batch_size = 64
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
td_errors_loss_fn = tf.compat.v1.losses.mean_squared_error 
gamma = 0.99

# Evaluation and Summary Parameters
num_eval_episodes = 100
eval_interval = 10000
log_interval = 1000
summary_interval = 1000
summaries_flush_secs = 10

# Results Directory and Run ID
run_id = 28421   # ID to differentiate between runs
root_dir = '~/'   # Has to be an existing directory
CS5Gamma = False

'''Argument Parser'''

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
        
global args
parser = argparse.ArgumentParser(description = 'DDPG Arguments')
parser.add_argument('--run_id', type = int, 
                    help = "Identifying integer for output folder names (default: 28421).")
parser.add_argument('--path', type = dir_path, 
                    help = "Path to existing directory for output (default: '~/').")
parser.add_argument('--use_CS5Gamma', type = bool, 
                    help = "Set true for CS5Gamma parameter configuration (default: False).")
args = parser.parse_args()
    
if args.run_id is not None:
    run_id = args.run_id
if args.use_CS5Gamma is not None:
    CS5Gamma = args.use_CS5Gamma
if args.path is not None:
    root_dir = args.path
       
if CS5Gamma:
    collect_steps_per_iteration = 5
    gamma = 0.995
    
'''Video Creation Method'''

def create_video(video_dir,    # Directory for video output
                 env_name,    # Environment Name for video creation
                 vid_policy,    # Policy to be used for video creation
                 video_id    # Identifying substring for folder names
                ):
    vid_py_env = gym.make(env_name)
    vid_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(vid_py_env))
    vid_py_env = wrappers.Monitor(vid_py_env, video_dir + "/" + str(video_id) + "/")
    time_step = vid_env.reset()
    policy_init = vid_policy.get_initial_state(vid_env.batch_size)
    policy_step = vid_policy.action(time_step, policy_init)
    vid_py_env.reset()
    score = 0
    while True:
        vid_py_env.render()
        action_tens = policy_step[0].numpy()
        next_state, reward, done, _ = vid_py_env.step(action_tens[0])
        score = score + reward
        if time_step.is_last():
            break
        if done:
            break
        time_step = vid_env.step(policy_step.action)
        policy_step = vid_policy.action(time_step, policy_step.state)
    vid_py_env.close()
    vid_env.close()
    return score

'''DDPG Algorithm'''

def DDPG_Bipedal(root_dir):
    
    # Setting up directories for results
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train' + '/' + str(run_id))
    eval_dir = os.path.join(root_dir, 'eval' + '/' + str(run_id))
    vid_dir = os.path.join(root_dir, 'vid' + '/' + str(run_id))
    
    # Set up Summary writer for training and evaluation
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis = summaries_flush_secs * 1000
    )
    train_summary_writer.set_as_default()
    
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis = summaries_flush_secs * 1000
    )
    eval_metrics = [
        # Metric to record average return
        tf_metrics.AverageReturnMetric(buffer_size = num_eval_episodes),
        # Metric to record average episode length
        tf_metrics.AverageEpisodeLengthMetric(buffer_size = num_eval_episodes)
    ]
    
    #Create global step
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % summary_interval, 0)):
        # Load Environment with different wrappers
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
        eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
        eval_py_env = suite_gym.load(env_name)
    
    
        # Define Actor Network
        actorNN = actor_network.ActorNetwork(
                  tf_env.time_step_spec().observation,
                  tf_env.action_spec(),
                  fc_layer_params=(400, 300),
        )
    
        # Define Critic Network
        NN_input_specs = (tf_env.time_step_spec().observation,
                          tf_env.action_spec()
        )
    
        criticNN = critic_network.CriticNetwork(
                   NN_input_specs,
                   observation_fc_layer_params = (400,),
                   action_fc_layer_params = None,
                   joint_fc_layer_params = (300,),
        )
        
        # Define & initialize DDPG Agent
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
                td_errors_loss_fn = tf.compat.v1.losses.mean_squared_error,
                gamma = gamma,
                train_step_counter = global_step
        )
        agent.initialize()
        
        # Determine which train metrics to display with summary writer
        train_metrics = [
                        tf_metrics.NumberOfEpisodes(),
                        tf_metrics.EnvironmentSteps(),
                        tf_metrics.AverageReturnMetric(),
                        tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        # Set policies for evaluation, initial collection
        eval_policy = agent.policy    # Actor policy
        collect_policy = agent.collect_policy    # Actor policy with OUNoise
        
        # Set up replay buffer
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        agent.collect_data_spec,
                        batch_size = tf_env.batch_size,
                        max_length = replay_buffer_capacity
        )
        
        # Define driver for initial replay buffer filling
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                                 tf_env,
                                 collect_policy,    # Initializes with random Parameters
                                 observers = [replay_buffer.add_batch],
                                 num_steps = initial_collect_steps
        )

        # Define collect driver for collect steps per iteration
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
            
        # Make 1000 random steps in tf_env and save in Replay Buffer
        logging.info(
            'Initializing replay buffer by collecting experience for 1000 steps with '
            'a random policy.', initial_collect_steps)
        initial_collect_driver.run()
        
        # Computes Evaluation Metrics
        results = metric_utils.eager_compute(
                  eval_metrics,
                  eval_tf_env,
                  eval_policy,
                  num_episodes = num_eval_episodes,
                  train_step = global_step,
                  summary_writer = eval_summary_writer,
                  summary_prefix = 'Metrics',
        )
        metric_utils.log_metrics(eval_metrics)
        
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0 

        # Dataset outputs steps in batches of 64
        dataset = replay_buffer.as_dataset(
                  num_parallel_calls = 3,
                  sample_batch_size = 64,
                  num_steps = 2).prefetch(3)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator) #Get experience from dataset (replay buffer)
            return agent.train(experience) #Train agent on that experience
        
        if use_tf_functions:
            train_step = common.function(train_step)
            
        
        for _ in range(num_iterations):
            start_time = time.time() # Get start time
            # Collect data for replay buffer
            time_step, policy_state = collect_driver.run(
                                      time_step = time_step,
                                      policy_state = policy_state,
            )
            # Train on experience 
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
                    name = 'iterations_per_sec', data = steps_per_sec, 
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
                metric_utils.log_metrics(eval_metrics)
                if results['AverageReturn'].numpy() >= 230.0:
                    video_score = create_video(
                                  video_dir = vid_dir,
                                  env_name = "BipedalWalker-v2",
                                  vid_policy = eval_policy,
                                  video_id = global_step.numpy()
                    )
    return train_loss

'''Run DDPG'''
DDPG_Bipedal(root_dir)

