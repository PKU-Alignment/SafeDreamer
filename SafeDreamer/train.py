import os
import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers
import datetime


def main(argv=None):
  from . import agent as agt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir_algo = config.logdir + now_time + '_' + str(config.method) + '_' + str(config.task) + '_' + str(config.seed)
  args = embodied.Config(
      **config.run, logdir=logdir_algo, use_cost=config.use_cost,
      batch_steps=config.batch_size * config.batch_length)
  print(config)
  os.environ['CUDA_VISIBLE_DEVICES'] = str(config.jax.logical_gpus)

  logdir = embodied.Path(logdir_algo)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  lag = PIDLagrangian(config)
  try:

    if args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args, lag)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args, lag)

    elif args.script == 'train_holdout':
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = make_envs(config, mode='eval')  # mode='eval'
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args, lag)

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ], multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_envs(config, **overrides):
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
  suite, task = config.task.split('_', 1)
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'safetygym': 'embodied.envs.safetygym:SafetyGym',
      'safetygymcoor': 'embodied.envs.safetygymcoor:SafetyGymCoor',
      'safetygymmujoco': 'embodied.envs.safetygym_mujoco:SafetyGymMujoco',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  kwargs.update({'platform':config.jax.platform})
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env

from collections import deque

import numpy as np
# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class PIDLagrangian():  # noqa: B024
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        config,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = config.pid.kp
        self._pid_ki: float = config.pid.ki
        self._pid_kd: float = config.pid.kd
        self._pid_d_delay = config.pid.d_delay
        self._pid_delta_p_ema_alpha: float = config.pid.delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = config.pid.delta_d_ema_alpha
        self._penalty_max: int = config.pid.penalty_max
        self._sum_norm: bool = config.pid.sum_norm
        self._diff_norm: bool = config.pid.diff_norm
        self._pid_i: float = config.pid.lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._pid_d: float = 0.0
        self._cost_limit: float = config.cost_limit
        self._cost_penalty: float = config.pid.init_penalty
        self._use_cost_decay: bool = config.pid.use_cost_decay
        self._current_cost_limit: float = config.pid.init_cost_limit
        if self._use_cost_decay:
          self._steps = [config.pid.decay_time_step * (i + 1) for i in range(config.pid.decay_num)]
          self._limits = [max(config.pid.init_cost_limit - i * config.pid.decay_limit_step,  config.cost_limit) for i in range(config.pid.decay_num)]
    @property
    def lagrange_penalty(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    @property
    def delta_p(self) -> float:
        """The lagrangian multiplier p."""
        return self._delta_p
    @property
    def pid_i(self) -> float:
        """The lagrangian multiplier i."""
        return self._pid_i

    @property
    def pid_d(self) -> float:
        """The lagrangian multiplier d."""
        return self._pid_d


    def pid_update(self, epcost, step) -> None:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        ep_cost_avg = epcost
        if self._use_cost_decay:
          for i, threshold in enumerate(self._steps):
            if step < threshold:
              self._current_cost_limit = self._limits[i]
              break
          else:
            self._current_cost_limit = self._cost_limit
        else:
          self._current_cost_limit = self._cost_limit

        delta = float(ep_cost_avg - self._current_cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * self._pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = np.clip(self._cost_penalty, 0.0, self._penalty_max)
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p


if __name__ == '__main__':
  main()
