import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from sac_models import StochasticActor, Critic, SAC
from samplers import Sampler
from buffers import ReplayBuffer
from envs.envs import (ExpertInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       CustomReacher2Env, CustomReacher3Env, ExpertHalfCheetahNCEnv,
                       DMCartPoleBalanceEnv, DMCartPoleSwingUpEnv, DMPendulumEnv)
from envs.maze_envs import CustomPointUMazeSize3Env
import time


def train_expert(env_name):
    """
    Train expert policy in an environment.

    Parameters
    ----------
    env_name : Source environment to collect the demonstrations.
    """

    # Environments
    if env_name == 'InvertedPendulum-v2':
        env = ExpertInvertedPendulumEnv()
        episode_limit = 1000
    elif env_name == 'InvertedDoublePendulum-v2':
        env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 1000
    elif env_name == 'Reacher2-v2':
        env = CustomReacher2Env(l2_penalty=True)
        episode_limit = 50
    elif env_name == 'Reacher3-v2':
        env = CustomReacher3Env(l2_penalty=True)
        episode_limit = 50
    elif env_name == 'HalfCheetah-v2':
        env = ExpertHalfCheetahNCEnv()
        episode_limit = 1000
    elif env_name == 'PointUMaze-v2':
        env = CustomPointUMazeSize3Env()
        episode_limit = 1000
    elif env_name == 'DMCartPoleBalance':
        env = DMCartPoleBalanceEnv(track_camera=True)
        episode_limit = 1000
    elif env_name == 'DMCartPoleSwingUp':
        env = DMCartPoleSwingUpEnv()
        episode_limit = 1000
    elif env_name == 'DMPendulum':
        env = DMPendulumEnv()
        episode_limit = 1000
    else:
        raise NotImplementedError("Invalid env_name")

    # RL parameters
    buffer_size = 1000000
    init_random_samples = 1000
    exploration_noise = 0.2
    learning_rate = 1e-4
    batch_size = 256
    epochs = 200
    if env_name in ['PointUMaze-v2']:
        epochs = 20
    steps_per_epoch = 5000
    updates_per_step = 1
    update_actor_every = 1
    start_training = 512
    gamma = 0.99
    polyak = 0.995
    entropy_coefficient = 0.2
    clip_actor_gradients = False
    visual_env = False
    action_size = env.action_space.shape[0]
    tune_entropy_coefficient = True
    target_entropy = -1 * action_size

    # Models
    def make_actor():
        actor = StochasticActor([tf.keras.layers.Dense(256, 'relu'),
                                 tf.keras.layers.Dense(256, 'relu'),
                                 tf.keras.layers.Dense(action_size * 2)])
        return actor

    def make_critic():
        critic = Critic([tf.keras.layers.Dense(256, 'relu'),
                         tf.keras.layers.Dense(256, 'relu'),
                         tf.keras.layers.Dense(1)])
        return critic

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    sampler = Sampler(env, episode_limit=episode_limit,
                      init_random_samples=init_random_samples, visual_env=visual_env)
    agent = SAC(make_actor,
                make_critic,
                make_critic,
                actor_optimizer=optimizer,
                critic_optimizer=optimizer,
                gamma=gamma,
                polyak=polyak,
                entropy_coefficient=entropy_coefficient,
                tune_entropy_coefficient=tune_entropy_coefficient,
                target_entropy=target_entropy,
                clip_actor_gradients=clip_actor_gradients)
    if visual_env:
        obs = np.expand_dims(env.reset(), axis=0)
    else:
        obs = np.expand_dims(env.reset(), axis=0)
    agent(obs)
    agent.summary()

    # Training loop
    mean_test_returns = []
    mean_test_std = []
    steps = []

    step_counter = 0
    start_time = time.time()
    for e in range(epochs):
        while step_counter < (e + 1) * steps_per_epoch:
            traj_data = sampler.sample_trajectory(agent, exploration_noise, get_ims=False)
            replay_buffer.add(traj_data)
            if step_counter > start_training:
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * traj_data['n'],
                            act_delay=update_actor_every)
            step_counter += traj_data['n']
        print()
        print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
        print('Total time: {:8.1f} seconds'.format(time.time() - start_time))
        out = sampler.evaluate(agent, 10, get_ims=False)
        mean_test_returns.append(out['mean'])
        mean_test_std.append(out['std'])
        steps.append(step_counter)

    # Plot a learning curve
    plt.errorbar(steps, mean_test_returns, mean_test_std)
    plt.title(env_name)
    plt.xlabel('steps')
    plt.ylabel('returns')
    plt.savefig(os.path.join('sac_expert', '{}.png'.format(env_name)))

    return agent
