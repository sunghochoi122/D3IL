import os

from utils import log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv,
                       ExpertInvertedDoublePendulumEnv, AgentInvertedDoublePendulumEnv,
                       CustomReacher2Env, TiltedCustomReacher2Env,
                       CustomReacher3Env, TiltedCustomReacher3Env,
                       ExpertHalfCheetahNCEnv, LockedLegsHalfCheetahNCEnv,
                       DMCartPoleBalanceEnv, DMCartPoleSwingUpEnv, DMPendulumEnv, DMAcrobotEnv)
from envs.maze_envs import CustomPointUMazeSize3Env, CustomAntUMazeSize3Env
from samplers import Sampler
from utils import save_expert_trajectories


def collect_prior_data(realm_name, max_timesteps=10000, prior_samples_location='prior_data'):
    """
    Collect and save prior visual observations for an environment realm.

    Parameters
    ----------
    realm_name : Environment realm to collect the visual observations.
    max_timesteps : Maximum number of visual observations to collect, default is 10000.
    prior_samples_location : Folder to save the prior visual observations collected.
    """

    # Environments
    if realm_name == 'InvertedPendulum':
        prior_envs = [ExpertInvertedPendulumEnv(), AgentInvertedPendulumEnv(),
                      ExpertInvertedDoublePendulumEnv(), AgentInvertedDoublePendulumEnv()]
        prior_env_names = ['ExpertInvertedPendulum-v2', 'AgentInvertedPendulum-v2',
                           'ExpertInvertedDoublePendulum-v2', 'AgentInvertedDoublePendulum-v2']
        episode_limit = 50
    elif realm_name == 'Reacher':
        prior_envs = [CustomReacher2Env(), TiltedCustomReacher2Env(),
                      CustomReacher3Env(), TiltedCustomReacher3Env()]
        prior_env_names = ['Reacher2-v2', 'TiltedReacher2-v2',
                           'Reacher3-v2', 'TiltedReacher3-v2']
        episode_limit = 50
    elif realm_name == 'HalfCheetah':
        prior_envs = [ExpertHalfCheetahNCEnv(), LockedLegsHalfCheetahNCEnv()]
        prior_env_names = ['HalfCheetah-v2', 'LockedLegsHalfCheetah-v2']
        episode_limit = 200
    elif realm_name == 'UMaze':
        prior_envs = [CustomPointUMazeSize3Env(), CustomAntUMazeSize3Env()]
        prior_env_names = ['PointUMaze-v2', 'AntUMaze-v2']
        episode_limit = 1000
    elif realm_name == 'DMCartPoleBalance':
        prior_envs = [DMCartPoleBalanceEnv(track_camera=True)]
        prior_env_names = ['DMCartPoleBalance']
        episode_limit = 1000
    elif realm_name == 'DMCartPoleSwingUp':
        prior_envs = [DMCartPoleSwingUpEnv()]
        prior_env_names = ['DMCartPoleSwingUp']
        episode_limit = 1000
    elif realm_name == 'DMPendulum':
        prior_envs = [DMPendulumEnv()]
        prior_env_names = ['DMPendulum']
        episode_limit = 1000
    elif realm_name == 'DMAcrobot':
        prior_envs = [DMAcrobotEnv()]
        prior_env_names = ['DMAcrobot']
        episode_limit = 200
    else:
        raise NotImplementedError("Invalid realm_name")

    episodes_n = int(max_timesteps // episode_limit)

    # Collect and save data
    for env, env_name in zip(prior_envs, prior_env_names):
        saver_sampler = Sampler(env, episode_limit=episode_limit, init_random_samples=0, visual_env=True)
        if realm_name not in ['UMaze']:
            traj = saver_sampler.sample_test_trajectories(None, 0.0, episodes_n, False, get_ims=True)
        else:
            traj = saver_sampler.sample_test_steps(None, 0.0, max_timesteps, False, get_ims=True)
        log_trajectory_statistics(traj['ret'])

        os.makedirs(prior_samples_location + '/' + env_name, exist_ok=True)
        save_expert_trajectories(traj, env_name, prior_samples_location, visual_data=True)
    print('Prior trajectories successfully saved.')
