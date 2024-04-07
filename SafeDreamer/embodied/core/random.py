import numpy as np


class RandomAgent:

  def __init__(self, act_space):
    self.act_space = act_space

  def policy(self, obs, state=None, mode='train'):
    batch_size = len(next(iter(obs.values())))
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    act['log_entropy'] = np.zeros(batch_size)
    act['log_action_mean'] = np.zeros((batch_size, ) + self.act_space['action'].shape)
    act['log_action_std'] = np.zeros((batch_size, ) + self.act_space['action'].shape)
    act['log_plan_num_safe_traj'] = np.zeros(batch_size)
    act['log_plan_ret'] = np.zeros(batch_size)
    act['log_plan_cost'] = np.zeros(batch_size)
    act['log_lagrange_penalty'] = np.zeros(batch_size)
    return act, state
