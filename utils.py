import numpy as np


def log_trajectory_statistics(trajectory_rewards, log=True):
    """Log and return trajectory statistics."""
    out = {}
    out['n'] = len(trajectory_rewards)
    out['mean'] = np.mean(trajectory_rewards)
    out['max'] = np.max(trajectory_rewards)
    out['min'] = np.min(trajectory_rewards)
    out['std'] = np.std(trajectory_rewards)
    if log:
        print('Number of completed trajectories - {}'.format(out['n']))
        print('Latest trajectories mean reward - {}'.format(out['mean']))
        print('Latest trajectories max reward - {}'.format(out['max']))
        print('Latest trajectories min reward - {}'.format(out['min']))
        print('Latest trajectories std reward - {}'.format(out['std']))
    return out


def save_expert_trajectories(trajectories, env_name, file_location, visual_data=True):
    """Save full visual trajectories data."""
    np.save(file_location + '/' + env_name + '/expert_ims.npy', trajectories['ims'])
    np.save(file_location + '/' + env_name + '/expert_ids.npy', trajectories['ids'])


def load_expert_trajectories(env_name, file_location, visual_data=True, load_ids=True, max_demos=None):
    """Load full visual trajectories data."""
    if max_demos is None:
        out = {'ims': np.load(file_location + '/' + env_name + '/expert_ims.npy'),
               'ids': np.load(file_location + '/' + env_name + '/expert_ids.npy')
               }
    else:
        out = {'ims': np.load(file_location + '/' + env_name + '/expert_ims.npy')[:max_demos],
               'ids': np.load(file_location + '/' + env_name + '/expert_ids.npy')[:max_demos],
               }
    return out
