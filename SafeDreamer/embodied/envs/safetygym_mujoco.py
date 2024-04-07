import functools
import os

import embodied
import numpy as np


class SafetyGymMujoco(embodied.Env):

  def __init__(self, env, platform='gpu', repeat=1, obs_key='observation', render=False, size=(64, 64), mode='train', camera_name='vision'):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    import gymnasium
    import safety_gymnasium
    env = safety_gymnasium.make(env,render_mode='rgb_array', width=size[0], height=size[1])

    self._dmenv = env
    from . import from_gymnasium
    self._env = from_gymnasium.FromGymnasium(self._dmenv,obs_key=obs_key)
    self._render = render if mode=='train' else True
    self._size = size
    self._repeat = repeat

  @property
  def repeat(self):
    return self._repeat

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    action = action.copy()
    if action['reset']:
      obs = self._reset()
    else:
        reward = 0.0
        cost = 0.0
        for i in range(self._repeat):
            obs = self._env.step(action)
            reward += obs['reward']
            if "cost" in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if "cost" in obs.keys():
            obs['cost'] = np.float32(cost)
    if self._render:
      obs['image'] = self.render()
    return obs

  def _reset(self):
    obs = self._env.step({'reset': True})
    return obs

  def render(self):
    return self._dmenv.render()
