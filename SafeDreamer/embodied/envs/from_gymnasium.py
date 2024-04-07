import functools

import embodied
import numpy as np

import gymnasium as gym

class FromGymnasium(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self.cost = 0
    self.cost_vases_contact = 0
    self.cost_vases_velocity = 0
    self.cost_hazards = 0
    self.cost_gremlins = 0
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


  def initial_reset(self):
    obs, info = self._env.reset(seed=0)
    self._done = True
    self._info = None
    return self._obs(obs, 0.0, 0.0, is_first=True)

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, info = self._env.reset()
      return self._obs(obs, 0.0, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    # if action[0] < 0.0:
      # action[0] = 0.0
    obs, reward, cost, terminated, truncated, self._info = self._env.step(action)
    if 'cost_vases_contact' in self._info.keys():
      self.cost_vases_contact += self._info['cost_vases_contact']
    if 'cost_vases_velocity' in self._info.keys():
      self.cost_vases_velocity += self._info['cost_vases_velocity']
    if 'cost_hazards' in self._info.keys():
      self.cost_hazards += self._info['cost_hazards']
    if 'cost_gremlins' in self._info.keys():
      self.cost_gremlins += self._info['cost_gremlins']
    self.cost += cost
    self._done = terminated or truncated
    if self._done:
      self.cost = 0
      self.cost_vases_contact = 0
      self.cost_vases_velocity = 0
      self.cost_hazards = 0
      self.cost_gremlins = 0
    return self._obs(
        obs, reward, cost,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

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
