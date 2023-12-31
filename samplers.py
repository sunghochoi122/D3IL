import numpy as np
import copy
from utils import log_trajectory_statistics


class Sampler(object):
    """Sampler to collect and evaluate agent behavior."""

    def __init__(self, env, episode_limit=1000, init_random_samples=1000, visual_env=False):
        """
        Parameters
        ----------
        env : Environment to run the agent.
        episode_limit : Maximum number of timesteps per trajectory, default is 1000.
        init_random_samples : Number of initial timesteps to execute random behavior, default is 1000.
        visual_env : Environment returns visual observations, default is False.
        """
        self._env = env
        self._eval_env = copy.deepcopy(self._env)
        self._visual_env = visual_env
        self._el = episode_limit
        self._nr = init_random_samples
        self._tc = 0
        self._ct = 0

        self._ob = None
        self._reset = True

    def _handle_ob(self, ob):
        if self._visual_env:
            return ob['obs']
        return ob

    def sample_steps(self, policy, noise_stddev, n_steps=1, dac_augmentation=False):
        """Collect a number of transition steps with policy."""
        obs, nobs, acts, rews, dones = [], [], [], [], []
        if self._visual_env:
            visual_obs = []
        for i in range(n_steps):
            if self._reset or self._ct >= self._el:
                self._ct = 0
                self._reset = False
                self._ob = self._env.reset()
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = np.array(policy.get_action(np.expand_dims(self._ob.astype('float32'), axis=0), noise_stddev))[0]
            obs.append(self._ob)
            acts.append(act)
            self._ob, rew, self._reset, info = self._env.step(act)
            if self._visual_env:
                visual_obs.append(self._env.get_ims())
            nobs.append(self._ob)
            rews.append(rew)
            dones.append(self._reset)
            self._ct += 1
            self._tc += 1
            if dac_augmentation:
                if self._reset:
                    nobs[-1] = self._env.absorbing_state
                    dones[-1] = False
                    obs.append(self._env.absorbing_state)
                    nobs.append(self._env.absorbing_state)
                    acts.append(np.zeros(self._env.action_space.shape))
                    rews.append(0.0)
                    dones.append(False)
                    self._ct += 1
        self._reset = True
        out = {'obs': np.stack(obs),
               'nobs': np.stack(nobs),
               'act': np.stack(acts),
               'rew': np.array(rews),
               'don': np.array(dones),
               'n': n_steps
               }
        if self._visual_env:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_trajectory(self, policy, noise_stddev, dac_augmentation=False, get_ims=True):
        """Collect a full trajectory with policy."""
        obs, nobs, acts, rews, dones = [], [], [], [], []
        if self._visual_env and get_ims:
            visual_obs = []
        ret = 0
        ct = 0
        done = False
        ob = self._env.reset()
        if self._visual_env and get_ims:
            _ = self._env.get_ims()
        while not done and ct < self._el:
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = np.array(policy.get_action(np.expand_dims(ob.astype('float32'), axis=0), noise_stddev))[0]
            obs.append(ob)
            acts.append(act)
            ob, rew, done, info = self._env.step(act)
            if self._visual_env and get_ims:
                visual_obs.append(self._env.get_ims())
            nobs.append(ob)
            rews.append(rew)
            dones.append(done)
            ret += rew
            ct += 1
            self._tc += 1
            if dac_augmentation:
                if done:
                    nobs[-1] = self._env.absorbing_state
                    dones[-1] = False
                    obs.append(self._env.absorbing_state)
                    nobs.append(self._env.absorbing_state)
                    acts.append(np.zeros(self._env.action_space.shape))
                    rews.append(0.0)
                    dones.append(False)
                    ct += 1
        out = {'obs': np.stack(obs),
               'nobs': np.stack(nobs),
               'act': np.stack(acts),
               'rew': np.array(rews),
               'don': np.array(dones),
               'n': ct,
               'ret': ret
               }
        if self._visual_env and get_ims:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_test_trajectories(self, policy, noise_stddev, n=5, visualize=False, only_visual_data=False, get_ims=False):
        """Collect multiple trajectories with policy keeping track of trajectory-specific statistics."""
        obs, nobs, acts, rews, dones, rets, ids = [], [], [], [], [], [], []
        if policy is None:
            print('WARNING: running random policy')
        if self._visual_env and get_ims:
            visual_obs = []
        for i in range(n):
            ret = 0
            ct = 0
            done = False
            ob = self._eval_env.reset()
            if self._visual_env and get_ims:
                _ = self._eval_env.get_ims()
            while not done and ct < self._el:
                if policy is not None:
                    act = np.array(policy.get_action(np.expand_dims(ob.astype('float32'), axis=0), noise_stddev))[0]
                else:
                    act = self._eval_env.action_space.sample()
                obs.append(ob)
                acts.append(act)
                ob, rew, done, info = self._eval_env.step(act)
                if visualize:
                    self._eval_env.render()
                if self._visual_env and get_ims:
                    visual_obs.append(self._eval_env.get_ims())
                nobs.append(ob)
                rews.append(rew)
                dones.append(done)
                ids.append(i)
                ret += rew
                ct += 1
            rets.append(ret)
        out = {'obs': np.stack(obs),
               'nobs': np.stack(nobs),
               'act': np.stack(acts),
               'rew': np.array(rews),
               'don': np.array(dones),
               'n': ct,
               'ret': rets,
               'ids': np.array(ids)
               }
        if self._visual_env and get_ims:
            out['ims'] = np.stack(visual_obs)
        return out

    def evaluate(self, policy, n=10, log=True, get_ims=False):
        """Collect multiple trajectories with policy and log trajectory-specific statistics."""
        traj = self.sample_test_trajectories(policy, 0.0, n, get_ims=get_ims)
        return log_trajectory_statistics(traj['ret'], log)

    def sample_test_steps(self, policy, noise_stddev, n_steps=5, visualize=False, only_visual_data=False, get_ims=False):
        """Collect multiple trajectories with policy keeping track of trajectory-specific statistics."""
        obs, nobs, acts, rews, dones, rets, ids = [], [], [], [], [], [], []
        if policy is None:
            print('WARNING: running random policy')
        if self._visual_env and get_ims:
            visual_obs = []
        ret = 0
        ct = 0
        done = True
        nep = 0
        for i in range(n_steps):
            if done or ct >= self._el:
                ret = 0
                ct = 0
                done = False
                ob = self._eval_env.reset()
                if self._visual_env and get_ims:
                    _ = self._eval_env.get_ims()
            if policy is not None:
                act = np.array(policy.get_action(np.expand_dims(ob.astype('float32'), axis=0), noise_stddev))[0]
            else:
                act = self._eval_env.action_space.sample()
            obs.append(ob)
            acts.append(act)
            ob, rew, done, info = self._eval_env.step(act)
            if visualize:
                self._eval_env.render()
            if self._visual_env and get_ims:
                visual_obs.append(self._eval_env.get_ims())
            nobs.append(ob)
            rews.append(rew)
            dones.append(done)
            ids.append(nep)
            ret += rew
            ct += 1
            if done or ct >= self._el:
                rets.append(ret)
                nep += 1
        out = {'obs': np.stack(obs),
               'nobs': np.stack(nobs),
               'act': np.stack(acts),
               'rew': np.array(rews),
               'don': np.array(dones),
               'n': n_steps,
               'ret': rets,
               'ids': np.array(ids)
               }
        if self._visual_env and get_ims:
            out['ims'] = np.stack(visual_obs)
        return out
