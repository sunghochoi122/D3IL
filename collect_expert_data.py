import os

from utils import log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, ExpertInvertedDoublePendulumEnv,
                       CustomReacher2Env, CustomReacher3Env, ExpertHalfCheetahNCEnv,
                       DMCartPoleBalanceEnv, DMCartPoleSwingUpEnv, DMPendulumEnv)
from envs.maze_envs import CustomPointUMazeSize3Env
from samplers import Sampler
from utils import save_expert_trajectories


def collect_expert_data(agent, env_name, max_timesteps=10000, expert_samples_location='expert_data'):
    """
    Collect and save demonstrations with trained expert agent.

    Parameters
    ----------
    agent : Trained expert agent.
    env_name : Source environment to collect the demonstrations.
    max_timesteps : Maximum number of visual observations to collect, default is 10000.
    expert_samples_location : Folder to save the expert demonstrations collected.
    """

    # Environments
    if env_name == 'InvertedPendulum-v2':
        expert_env = ExpertInvertedPendulumEnv()
        episode_limit = 1000
    elif env_name == 'InvertedDoublePendulum-v2':
        expert_env = ExpertInvertedDoublePendulumEnv()
        episode_limit = 1000
    elif env_name == 'Reacher2-v2':
        expert_env = CustomReacher2Env()
        episode_limit = 50
    elif env_name == 'Reacher3-v2':
        expert_env = CustomReacher3Env()
        episode_limit = 50
    elif env_name == 'HalfCheetah-v2':
        expert_env = ExpertHalfCheetahNCEnv()
        episode_limit = 200
    elif env_name == 'PointUMaze-v2':
        expert_env = CustomPointUMazeSize3Env()
        episode_limit = 1000
    elif env_name == 'DMCartPoleBalance':
        expert_env = DMCartPoleBalanceEnv(track_camera=True)
        episode_limit = 1000
    elif env_name == 'DMCartPoleSwingUp':
        expert_env = DMCartPoleSwingUpEnv()
        episode_limit = 1000
    elif env_name == 'DMPendulum':
        expert_env = DMPendulumEnv()
        episode_limit = 1000
    else:
        raise NotImplementedError("Invalid env_name")
    episodes_n = int(max_timesteps // episode_limit)

    saver_sampler = Sampler(expert_env, episode_limit=episode_limit, init_random_samples=0, visual_env=True)

    # Collect demonstrations
    if env_name not in ['PointUMaze-v2']:
        traj = saver_sampler.sample_test_trajectories(agent, 0.0, episodes_n, False, get_ims=True)
    else:
        traj = saver_sampler.sample_test_steps(agent, 0.0, max_timesteps, False, get_ims=True)
    print()
    log_trajectory_statistics(traj['ret'])

    # Save demonstrations
    os.makedirs(expert_samples_location + '/' + env_name)
    save_expert_trajectories(traj, env_name, expert_samples_location, visual_data=True)
    print('Expert trajectories successfully saved.')
