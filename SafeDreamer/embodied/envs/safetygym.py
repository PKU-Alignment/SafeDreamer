import functools
import os

import embodied
import numpy as np
import cv2


class SafetyGym(embodied.Env):

  def __init__(self, env, platform='gpu', repeat=1, obs_key='image', render=False, size=(64, 64), camera=-1, mode='train', camera_name='vision'):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    import gymnasium
    import safety_gymnasium
    if mode=='train':
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=size[0], height=size[1])
    elif mode=='eval':
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=1024, height=1024)

    self._dmenv = env
    from . import from_gymnasium
    self._env = from_gymnasium.FromGymnasium(self._dmenv,obs_key=obs_key)
    self._render = render if mode=='train' else True
    self._size = size
    self._camera = camera
    self._camera_name = camera_name
    self._repeat = repeat
    self._mode = mode
  @property
  def repeat(self):
    return self._repeat

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
      if self._camera_name == 'vision_front_back':
        spaces['image2'] = embodied.Space(np.uint8, self._size + (3,))

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
            if 'cost' in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if 'cost' in obs.keys():
            obs['cost'] = np.float32(cost)
    #obs= obs['vision']
    if self._render:
      if self._mode == 'train':
        image1 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision', cost={})
        obs['image'] = image1
        if self._camera_name == 'vision_front_back':
          image2 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision_back', cost={})
          obs['image2'] = image2
      elif self._mode == 'eval':
        obs['image_orignal'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='vision', cost={})
        image = cv2.resize(
            obs['image_orignal'], self._size, interpolation=cv2.INTER_AREA)
        obs['image'] = image
        obs['image_far'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='fixedfar', cost={'cost_sum': obs['cost']})

        if self._camera_name == 'vision_front_back':
          obs['image_orignal2'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='vision_back', cost={})
          image2 = cv2.resize(
              obs['image_orignal2'], self._size, interpolation=cv2.INTER_AREA)
          obs['image2'] = image2

    return obs

  def _reset(self):
    obs = self._env.step({'reset': True})
    return obs

  def render(self):
    return self._dmenv.render()
