import cv2
import numpy as np
import os
from dmc2gym.wrappers import DMCWrapper, _flatten_obs
from gym import utils
from gym.envs.mujoco import mujoco_env

# ====================================================================================================
class _FrameBufferEnv:
    def __init__(self, past_frames):
        self._initialized = False
        self._past_frames = past_frames

    def _init_buffer(self, im):
        self._im_size = im.shape
        self._reset_buffer()

    def _reset_buffer(self, ):
        self._frames_buffer = np.zeros([self._past_frames] + list(self._im_size)).astype('uint8')

    def _update_buffer(self, im):
        self._frames_buffer = np.concatenate([np.expand_dims(im.astype('uint8'), 0),
                                              self._frames_buffer[:-1, :, :, :]], axis=0).astype('uint8')


# ====================================================================================================
class _CustomInvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, size=(32, 32), color_permutation=[0, 1, 2],
                 smoothing_factor=0.0, past_frames=4, not_done=True):
        self._size = size
        self._not_done = not_done
        self._color_permutation = color_permutation
        self._smooth = 1.0 - smoothing_factor
        _FrameBufferEnv.__init__(self, past_frames)
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/inverted_pendulum.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        if done and self._not_done:
            done = False
            reward = 0.0
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def get_ims(self):
        raw_im = (self.render(mode='rgb_array'))[:, :, self._color_permutation] * self._smooth
        im = cv2.resize(raw_im, dsize=self._size, interpolation=cv2.INTER_AREA)
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class AgentInvertedPendulumEnv(_CustomInvertedPendulumEnv):
    def __init__(self, ):
        super(AgentInvertedPendulumEnv, self).__init__(size=(32, 32),
                                                       color_permutation=[0, 1, 2],
                                                       smoothing_factor=0.0,
                                                       past_frames=4,
                                                       not_done=True)


class ExpertInvertedPendulumEnv(_CustomInvertedPendulumEnv):
    def __init__(self, ):
        super(ExpertInvertedPendulumEnv, self).__init__(size=(32, 32),
                                                        color_permutation=[2, 1, 0],
                                                        smoothing_factor=0.1,
                                                        past_frames=4,
                                                        not_done=True)


class _CustomInvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, size=(32, 32), color_permutation=[0, 1, 2],
                 smoothing_factor=0.0, past_frames=4, not_done=True):
        self._size = size
        self._not_done = not_done
        self._failure = False
        self._color_permutation = color_permutation
        self._smooth = 1.0 - smoothing_factor
        _FrameBufferEnv.__init__(self, past_frames)
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/inverted_double_pendulum.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        if done and self._not_done:
            done = False
            self._failure = True
        if self._failure:
            r = 0.0
        return ob, r, done, {}

    def reset_model(self):
        self._failure = False
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],
            np.sin(self.sim.data.qpos[1:]),
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def get_ims(self):
        raw_im = (self.render(mode='rgb_array'))[:, :, self._color_permutation] * self._smooth
        im = cv2.resize(raw_im, dsize=self._size, interpolation=cv2.INTER_AREA)
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class AgentInvertedDoublePendulumEnv(_CustomInvertedDoublePendulumEnv):
    def __init__(self, ):
        super(AgentInvertedDoublePendulumEnv, self).__init__(size=(32, 32),
                                                             color_permutation=[0, 1, 2],
                                                             smoothing_factor=0.0,
                                                             past_frames=4,
                                                             not_done=True)


class ExpertInvertedDoublePendulumEnv(_CustomInvertedDoublePendulumEnv):
    def __init__(self, ):
        super(ExpertInvertedDoublePendulumEnv, self).__init__(size=(32, 32),
                                                              color_permutation=[2, 1, 0],
                                                              smoothing_factor=0.1,
                                                              past_frames=4,
                                                              not_done=True)


# ====================================================================================================
class _CustomReacher2Env(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, l2_penalty=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/custom_reacher2.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        if self._l2_penalty:
            reward_ctrl = - np.mean(np.square(a)) * 2
        else:
            reward_ctrl = 0.0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        seed = np.random.randint(16)
        mag, ang = 0.15 + 0.05 * divmod(seed, 8)[0], divmod(seed, 8)[1] * np.pi / 4.0
        self.goal = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class TiltedCustomReacher2Env(_CustomReacher2Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(TiltedCustomReacher2Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class CustomReacher2Env(_CustomReacher2Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(CustomReacher2Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class _CustomReacher3Env(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, l2_penalty=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._l2_penalty = l2_penalty
        utils.EzPickle.__init__(self)
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/custom_reacher3.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        if self._l2_penalty:
            reward_ctrl = - np.mean(np.square(a)) * 2
        else:
            reward_ctrl = 0.0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        seed = np.random.randint(16)
        mag, ang = 0.15 + 0.05 * divmod(seed, 8)[0], divmod(seed, 8)[1] * np.pi / 4.0
        self.goal = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[3:],
            self.sim.data.qvel.flat[:3],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(48, 48), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class CustomReacher3Env(_CustomReacher3Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(CustomReacher3Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class TiltedCustomReacher3Env(_CustomReacher3Env):
    def __init__(self, past_frames=4, l2_penalty=False):
        super(TiltedCustomReacher3Env, self).__init__(past_frames=past_frames, l2_penalty=l2_penalty)

    def viewer_setup(self):
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


# ====================================================================================================
class _CustomHalfCheetahNCEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, action_penalties=True, reward_xmove_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self._action_penalties = action_penalties
        self._reward_xmove_only = reward_xmove_only
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/half_cheetah_nc.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return _CustomHalfCheetahNCEnv(reward_xmove_only=self._reward_xmove_only)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if not self._reward_xmove_only:
            if self._action_penalties:
                reward_ctrl = - 0.1 * np.square(action).sum()
            else:
                reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        else:
            reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class ExpertHalfCheetahNCEnv(_CustomHalfCheetahNCEnv):
    def __init__(self, past_frames=4, reward_xmove_only=False):
        super(ExpertHalfCheetahNCEnv, self).__init__(past_frames=past_frames, reward_xmove_only=reward_xmove_only)


class _CustomLLHalfCheetahNCEnv(mujoco_env.MujocoEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, action_penalties=True, reward_xmove_only=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self._action_penalties = action_penalties
        self._reward_xmove_only = reward_xmove_only
        path_to_xml = os.path.join(os.path.dirname(__file__), 'assets/half_cheetah_locked_legs_nc.xml')
        mujoco_env.MujocoEnv.__init__(self, path_to_xml, 2)
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return _CustomLLHalfCheetahNCEnv(reward_xmove_only=self._reward_xmove_only)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if not self._reward_xmove_only:
            if self._action_penalties:
                reward_ctrl = - 0.1 * np.square(action).sum()
            else:
                reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        else:
            reward_ctrl = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self._initialized:
            self._reset_buffer()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class LockedLegsHalfCheetahNCEnv(_CustomLLHalfCheetahNCEnv):
    def __init__(self, past_frames=4, reward_xmove_only=False):
        super(LockedLegsHalfCheetahNCEnv, self).__init__(past_frames=past_frames, reward_xmove_only=reward_xmove_only)


# ====================================================================================================
class _DMCWrapper(DMCWrapper):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs,
        visualize_reward,
        from_pixels,
        height,
        width,
        camera_id,
        frame_skip,
        environment_kwargs,
        channels_first
    ):
        super(_DMCWrapper, self).__init__(domain_name=domain_name,
                                          task_name=task_name,
                                          task_kwargs=task_kwargs,
                                          visualize_reward=visualize_reward,
                                          from_pixels=from_pixels,
                                          height=height,
                                          width=width,
                                          camera_id=camera_id,
                                          frame_skip=frame_skip,
                                          environment_kwargs=environment_kwargs,
                                          channels_first=channels_first)

    def step(self, action):
        action = self._convert_action(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        if self._initialized:
            self._reset_buffer()
        return obs


# ====================================================================================================
class DMCartPoleBalanceEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, track_camera=False):
        _FrameBufferEnv.__init__(self, past_frames)
        self.track_camera = track_camera
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        super(DMCartPoleBalanceEnv, self).__init__(domain_name='cartpole',
                                                   task_name='balance_sparse',
                                                   task_kwargs=task_kwargs,
                                                   visualize_reward=False,
                                                   from_pixels=False,
                                                   height=64,
                                                   width=64,
                                                   camera_id=0,
                                                   frame_skip=1,
                                                   environment_kwargs=None,
                                                   channels_first=False
                                                   )
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return DMCartPoleBalanceEnv(track_camera=self.track_camera)

    def get_ims(self):
        if self.track_camera:
            self._physics.data.cam_xpos[0][0] = self._physics.get_state()[0]
        im = self.render(mode='rgb_array')[8:56, 8:56]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMCartPoleSwingUpEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4, track_camera=True):
        _FrameBufferEnv.__init__(self, past_frames)
        self.track_camera = track_camera
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMCartPoleSwingUpEnv, self).__init__(domain_name='cartpole',
                                                   task_name='swingup',
                                                   task_kwargs=task_kwargs,
                                                   visualize_reward=False,
                                                   from_pixels=False,
                                                   height=64,
                                                   width=64,
                                                   camera_id=0,
                                                   frame_skip=1,
                                                   environment_kwargs=None,
                                                   channels_first=False
                                                   )
        utils.EzPickle.__init__(self)

    def __deepcopy__(self, memodict={}):
        return DMCartPoleSwingUpEnv(track_camera=self.track_camera)

    def get_ims(self):
        if self.track_camera:
            self._physics.data.cam_xpos[0][0] = self._physics.get_state()[0]
        im = self.render(mode='rgb_array')[8:56, 8:56]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMPendulumEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMPendulumEnv, self).__init__(domain_name='pendulum',
                                            task_name='swingup',
                                            task_kwargs=task_kwargs,
                                            visualize_reward=False,
                                            from_pixels=False,
                                            height=64,
                                            width=64,
                                            camera_id=0,
                                            frame_skip=1,
                                            environment_kwargs=None,
                                            channels_first=False
                                            )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        self._physics.data.cam_xpos[0][2] = 0.6
        self._physics.data.cam_xmat[0] = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0])
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class DMAcrobotEnv(_DMCWrapper, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        self.past_frames = past_frames
        task_kwargs = {}
        task_kwargs['random'] = np.random.randint(0, 100000)
        task_kwargs['time_limit'] = 1000
        super(DMAcrobotEnv, self).__init__(domain_name='acrobot',
                                           task_name='swingup',
                                           task_kwargs=task_kwargs,
                                           visualize_reward=False,
                                           from_pixels=False,
                                           height=64,
                                           width=64,
                                           camera_id=0,
                                           frame_skip=1,
                                           environment_kwargs=None,
                                           channels_first=False
                                           )
        utils.EzPickle.__init__(self)

    def get_ims(self):
        im = self.render(mode='rgb_array')[4:60, 4:60]
        im = cv2.resize(im, dsize=(32, 32), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames
