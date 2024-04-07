import functools
import gymnasium as gym

import embodied
import numpy as np

import cv2

class Carracing(embodied.Env):
  def __init__(self, env, repeat=1, obs_key='image', act_key='action', render=False, size=(64, 64), mode='train'):

    # some bugs for metadrive run in cloud that should initialize an instance firstly
    name = env
    self._mode = mode
    if mode=='train':
      self._env = gym.make("CarRacing-v2")
    elif mode=='eval':
      self._env = gym.make("CarRacing-v2", render_mode='rgb_array')

    self._size = size
    self._repeat = repeat
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'cost': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  @property
  def task(self):
    return self._env.task


  def step_single(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset()

      return self._obs(obs, 0.0, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    # if action[0] < 0.0:
      # action[0] = 0.0
    obs, reward, terminated, truncated, self._info = self._env.step(action)

    cost = 0
    car = self._env.car
    for wheel in car.wheels:
        if wheel.skid_start is not None or wheel.skid_particle is not None:
            cost = 1
    self._done = terminated or truncated
    if self._done:
        print(self._info)
    return self._obs(
        obs, reward, cost,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])

    action = action.copy()
    if action['reset']:
      obs = self.step_single({'reset': True})
      obs['image_orignal'] = self._env.render()
      obs['image'] = cv2.resize(
          obs['image'], self._size, interpolation=cv2.INTER_AREA)
      return obs

    else:
        reward = 0.0
        cost = 0.0
        for i in range(self._repeat):
            obs = self.step_single(action)
            reward += obs['reward']
            if 'cost' in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if 'cost' in obs.keys():
            obs['cost'] = np.float32(cost)
    if self._mode == 'eval':
      obs['image_orignal'] = self._env.render()
    obs['image'] = cv2.resize(
        obs['image'], self._size, interpolation=cv2.INTER_AREA)

    return obs

  def _obs(
      self, obs, reward, cost, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        cost=np.float32(cost),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
