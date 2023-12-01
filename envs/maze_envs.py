import numpy as np
from gym import utils
import cv2
from envs.envs import _FrameBufferEnv
from envs.mujoco_maze.maze_env import MazeEnv
from envs.mujoco_maze.point import PointSize3Env
from envs.mujoco_maze.ant import AntSize3Env
from envs.mujoco_maze.maze_task import GoalRewardUMaze

"""
Environment based on
https://github.com/kngwyu/mujoco-maze
"""

class _CustomUMazeEnv(MazeEnv, utils.EzPickle, _FrameBufferEnv):
    def __init__(self, model_cls, maze_task, maze_size_scaling, inner_reward_scaling, past_frames=4):
        _FrameBufferEnv.__init__(self, past_frames)
        self._initialized = False
        utils.EzPickle.__init__(self)
        MazeEnv.__init__(self,
                         model_cls=model_cls,
                         maze_task=maze_task,
                         maze_size_scaling=maze_size_scaling,
                         inner_reward_scaling=inner_reward_scaling)

    def step(self, action: np.ndarray):
        self.t += 1
        if self.wrapped_env.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            old_objballs = self._objball_positions()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            new_objballs = self._objball_positions()
            # Checks that the new_position is in the wall
            collision = self._collision.detect(old_pos, new_pos)
            if collision is not None:
                pos = collision.point + self._restitution_coef * collision.rest()
                if self._collision.detect(old_pos, pos) is not None:
                    # If pos is also not in the wall, we give up computing the position
                    # print("Collision! Go to old position.")
                    self.wrapped_env.set_xy(old_pos)
                else:
                    # print("Collision! Go to modified position.")
                    self.wrapped_env.set_xy(pos)
            # Do the same check for object balls
            for name, old, new in zip(self.object_balls, old_objballs, new_objballs):
                collision = self._objball_collision.detect(old, new)
                if collision is not None:
                    pos = collision.point + self._restitution_coef * collision.rest()
                    if self._objball_collision.detect(old, pos) is not None:
                        pos = old
                    idx = self.wrapped_env.model.body_name2id(name)
                    self.wrapped_env.data.xipos[idx][:2] = pos
        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        inner_reward = self._inner_reward_scaling * inner_reward
        outer_reward = self._task.reward(next_obs)
        done = self._task.termination(next_obs)
        info["position"] = self.wrapped_env.get_xy()
        info["inner_reward"] = inner_reward
        info["outer_reward"] = outer_reward
        return next_obs, inner_reward + outer_reward, done, info

    def reset(self):
        self.t = 0
        self.wrapped_env.reset()
        # Samples a new goal
        if self._task.sample_goals():
            self.set_marker()
        # Samples a new start position
        if len(self._init_positions) > 1:
            xy = np.random.choice(self._init_positions)
            self.wrapped_env.set_xy(xy)

        if self._initialized:
            self._reset_buffer()
        return self._get_obs()


class CustomPointUMazeSize3Env(_CustomUMazeEnv):
    def __init__(self, ):
        super(CustomPointUMazeSize3Env, self).__init__(model_cls=PointSize3Env,
                                                       maze_task=GoalRewardUMaze,
                                                       maze_size_scaling=3,
                                                       inner_reward_scaling=GoalRewardUMaze.INNER_REWARD_SCALING,
                                                       past_frames=4)

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames


class CustomAntUMazeSize3Env(_CustomUMazeEnv):
    def __init__(self, ):
        super(CustomAntUMazeSize3Env, self).__init__(model_cls=AntSize3Env,
                                                     maze_task=GoalRewardUMaze,
                                                     maze_size_scaling=3,
                                                     inner_reward_scaling=GoalRewardUMaze.INNER_REWARD_SCALING,
                                                     past_frames=4)

    def get_ims(self):
        im = self.render(mode='rgb_array')
        im = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_AREA).astype('int32')
        if not self._initialized:
            self._init_buffer(im)
            self._initialized = True
        self._update_buffer(im)
        curr_frames = self._frames_buffer.astype('uint8')
        return curr_frames
