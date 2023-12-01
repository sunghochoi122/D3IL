import tensorflow as tf
import numpy as np

from sac_models import StochasticActor, Critic, SAC
from envs.envs import (ExpertInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       CustomReacher2Env, CustomReacher3Env, ExpertHalfCheetahNCEnv,
                       DMCartPoleBalanceEnv, DMCartPoleSwingUpEnv, DMPendulumEnv)
from envs.maze_envs import CustomPointUMazeSize3Env


def make_dummy_expert(env_name):
    """
    Make a dummy model to load the expert

    Parameters
    ----------
    env_name : Source environment to collect the demonstrations.
    """

    # Environments
    if env_name == 'InvertedPendulum-v2':
        env = ExpertInvertedPendulumEnv()
    elif env_name == 'InvertedDoublePendulum-v2':
        env = ExpertInvertedDoublePendulumEnv()
    elif env_name == 'Reacher2-v2':
        env = CustomReacher2Env()
    elif env_name == 'Reacher3-v2':
        env = CustomReacher3Env()
    elif env_name == 'HalfCheetah-v2':
        env = ExpertHalfCheetahNCEnv()
    elif env_name == 'PointUMaze-v2':
        env = CustomPointUMazeSize3Env()
    elif env_name == 'DMCartPoleBalance':
        env = DMCartPoleBalanceEnv()
    elif env_name == 'DMCartPoleSwingUp':
        env = DMCartPoleSwingUpEnv()
    elif env_name == 'DMPendulum':
        env = DMPendulumEnv()
    else:
        raise NotImplementedError("Invalid env_name")

    # RL parameters
    learning_rate = 3e-4
    gamma = 0.99
    polyak = 0.995
    entropy_coefficient = 0.2
    clip_actor_gradients = False
    action_size = env.action_space.shape[0]
    tune_entropy_coefficient = True
    target_entropy = -1*action_size

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
    obs = np.expand_dims(env.reset(), axis=0)
    agent(obs)
    agent.summary()
    return agent
