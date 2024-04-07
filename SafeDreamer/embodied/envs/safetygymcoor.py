import functools
import os

import embodied
import numpy as np


class SafetyGymCoor(embodied.Env):

  def __init__(self, env, platform='gpu', repeat=1, obs_key='observation', render=False, size=(64, 64), camera=-1, mode='train', camera_name='vision'):

    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'


    import gymnasium
    import safety_gymnasium
    if mode=='train':
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name='fixedfar', width=64, height=64)
    elif mode=='eval':
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name='fixedfar', width=1024, height=1024)
    self._mode = mode
    self._dmenv = env
    from . import from_gymnasium
    self._env = from_gymnasium.FromGymnasium(self._dmenv,obs_key=obs_key)
    self._repeat = repeat
    self._render = render if mode=='train' else True
    self._size = size
    self._camera = camera

    self._constraints = ['hazards']  #'gremlins', 'buttons'],
    self._xyz_sensors = ['velocimeter', 'accelerometer']
    self._angle_sensors = ['gyro', 'magnetometer']
    self._flatten_order = (
        self._xyz_sensors
        + self._angle_sensors
        + ['goal']
        + self._constraints
        + ['robot_m']
        + ['robot']
    )
    self._base_state = self._xyz_sensors + self._angle_sensors
    self._env.initial_reset()
    self.goal_position = self._env.task.goal.pos
    self.robot_position = self._env.task.agent.pos
    self.hazards_position = self._env.task.hazards.pos
    self.goal_distance = self._dist_xy(self.robot_position, self.goal_position)
    coordinate_sensor_obs = self._get_coordinate_sensor()
    self._coordinate_obs_size = sum(
        np.prod(i.shape) for i in list(coordinate_sensor_obs.values())
    )
    offset = 0
    self.key_to_slice = {}
    for k in self._flatten_order:
        k_size = np.prod(coordinate_sensor_obs[k].shape)
        self.key_to_slice[k] = slice(offset, offset + k_size)
        print("key_to_slice", k, self.key_to_slice[k], coordinate_sensor_obs[k])
        offset += k_size

    self._num_lidar_bin = 16
    self._max_lidar_dist = 3
    self.hazards_size = 0.2
    self.goal_size = 0.3
    #self.original_observation_space = self.observation_space
    self.coordinate_observation_space = gymnasium.spaces.Box(
        -np.inf,
        np.inf,
        (self._coordinate_obs_size,),
        dtype=np.float32,
    )
    #flat_coordinate_obs = self._get_flat_coordinate(coordinate_sensor_obs)
    # self.lidar_observation_space = gymnasium.spaces.Box(
    #     -np.inf,
    #     np.inf,
    #     (self.get_lidar_from_coordinate(flat_coordinate_obs).shape[0],),
    #     dtype=np.float32,
    # )

  @property
  def repeat(self):
    return self._repeat

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    spaces2 = {'observation': self.coordinate_observation_space}
    spaces2 = {k: self._convert(v) for k, v in spaces2.items()}
    spaces.update(spaces2)
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
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
    coordinate_sensor_obs = self._get_coordinate_sensor()
    obs_coor = self._get_flat_coordinate(coordinate_sensor_obs)
    obs['observation'] = obs_coor
    if self._render:
        if self._mode == 'eval':
            obs['image_far'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='fixedfar', cost={'cost_sum': obs['cost']})
        else:
            obs['image_far'] = self.render()
    return obs

  def _reset(self):
    obs = self._env.step({'reset': True})
    return obs

  def render(self):
    return self._dmenv.render()

  def _dist_xy(
      self,
      pos1,
      pos2,
  ) -> float:
      """Return the distance from the robot to an XY position.

      Args:
          pos1 (np.ndarray | list): The first position.
          pos2 (np.ndarray | list): The second position.

      Returns:
          distance (float): The distance between the two positions.
      """
      pos1 = np.asarray(pos1)
      pos2 = np.asarray(pos2)
      if pos1.shape == (3,):
          pos1 = pos1[:2]
      if pos2.shape == (3,):
          pos2 = pos2[:2]
      return np.sqrt(np.sum(np.square(pos1 - pos2)))

  def _ego_xy(self, robot_matrix, robot_pos, pos):
      """Return the egocentric XY vector to a position from the robot.

      Args:
          robot_matrix (np.ndarray): 3x3 rotation matrix.
          robot_pos (np.ndarray): 2D robot position.
          pos (np.ndarray): 2D position.

      Returns:
          2D_egocentric_vector (np.ndarray): 2D egocentric vector.
      """
      assert pos.shape == (2,), f'Bad pos {pos}'
      assert robot_pos.shape == (2,), f'Bad robot_pos {robot_pos}'
      robot_3vec = robot_pos
      robot_mat = robot_matrix

      pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
      robot_3vec = np.concatenate([robot_3vec, [0]])
      world_3vec = pos_3vec - robot_3vec
      return np.matmul(world_3vec, robot_mat)[:2]

  def _get_flat_coordinate(self, coordinate_obs:dict) -> np.ndarray:
      """Get the flattened obs.

      Args:
          coordinate_obs: dict of coordinate and sensor observations.

      Returns:
          flat_obs (np.ndarray): flattened observation.
      """
      flat_obs = np.zeros(self.coordinate_observation_space.shape[0])
      for k in self._flatten_order:
          idx = self.key_to_slice[k]
          flat_obs[idx] = coordinate_obs[k].flat
      return flat_obs

  def _get_coordinate_sensor(self) -> dict:
      """
      Return the coordinate observation and sensor observation.
      We will ignore the z-axis coordinates in every poses.
      The returned obs coordinates are all in the robot coordinates.

      Returns:
          coordinate_obs (dict): coordinate observation.
      """
      obs = {}
      robot_matrix = self._env.task.agent.mat
      obs['robot_m'] = np.array(robot_matrix[0][:2])

      robot_pos = self._env.task.agent.pos
      goal_pos = self._env.task.goal.pos
      # vases_pos_list = self._env.task.vases.pos  # list of shape (3,) ndarray
      hazards_pos_list = self._env.task.hazards.pos  # list of shape (3,) ndarray
      # ego_goal_pos = self._env.task._ego_xy(goal_pos[:2])
      # [self._env.task._ego_xy(pos[:2]) for pos in vases_pos_list]  # list of shape (2,) ndarray
      # ego_hazards_pos_list = [
      #     self._env.task._ego_xy(pos[:2]) for pos in hazards_pos_list
      # ]  # list of shape (2,) ndarray

      ego_goal_pos = self._ego_xy(robot_matrix, robot_pos[:2], goal_pos[:2])
      ego_hazards_pos_list = [
          self._ego_xy(robot_matrix, robot_pos[:2], pos[:2]) for pos in hazards_pos_list
      ]  # list of shape (2,) ndarray

      # append obs to the dict
      for sensor in self._xyz_sensors:  # Explicitly listed sensors
          if sensor == 'accelerometer':
              obs[sensor] = self._env.task.agent.get_sensor(sensor)[:1]  # only x axis matters
          elif sensor == 'ballquat_rear':
              obs[sensor] = self._env.task.agent.get_sensor(sensor)
          else:
              obs[sensor] = self._env.task.agent.get_sensor(sensor)[:2]  # only x,y axis matters

      for sensor in self._angle_sensors:
          if sensor == 'gyro':
              obs[sensor] = self._env.task.agent.get_sensor(sensor)[
                  2:
              ]  # [2:] # only z axis matters
              # pass # gyro does not help
          else:
              obs[sensor] = self._env.task.agent.get_sensor(sensor)
      # --------modification-----------------
      obs['robot'] = np.array(robot_pos[:2])
      obs['hazards'] = np.array(ego_hazards_pos_list)  # (hazard_num, 2)
      obs['goal'] = ego_goal_pos  # (2,)
      # obs['vases'] = np.array(ego_vases_pos_list)  # (vase_num, 2)
      return obs

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)