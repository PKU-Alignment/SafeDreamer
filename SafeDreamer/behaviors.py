import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils

import jax
from jax import lax

tree_map = jax.tree_util.tree_map

sg = lambda x: tree_map(jax.lax.stop_gradient, x)

def cost_from_state(wm, state):
  recon = wm.heads['decoder'](state)
  recon = recon['observation'].mode()
  hazards_size = 0.25
  batch_size = recon.shape[0] * recon.shape[1]
  hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
  hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
      batch_size,
      -1,
  )
  condition = jnp.less_equal(hazards_dist, hazards_size)
  cost = jnp.where(condition, 1.0, 0.0)
  cost = cost.sum(1)
  # condition = jnp.greater_equal(cost, 1.0)
  # cost = jnp.where(condition, 1.0, 0.0)

  cost = cost.reshape(recon.shape[0], recon.shape[1])
  return cost

class Greedy(nj.Module):

  def __init__(self, wm, act_space, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    if config.use_cost:
      if config.use_cost_model:
        costfn = lambda s: wm.heads['cost'](s).mean()[1:]
      else:
        costfn = lambda s: cost_from_state(wm, s)[1:]
    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic')}
      if config.use_cost:
        cost_critics = {'extr': agent.CostVFunction(costfn, config, name='cost_critic')}
    else:
      raise NotImplementedError(config.critic_type)

    if config.use_cost:
      self.ac = agent.ImagSafeActorCritic(
          critics, cost_critics, {'extr': 1.0}, {'extr': 1.0}, act_space, config, name='safe_ac')
    else:
      self.ac = agent.ImagActorCritic(
          critics, {'extr': 1.0}, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}


class PIDPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.config = config
    self.ac = ac
    self.wm = wm
    self.act_space = act_space

    self.horizon = self.config.planner.horizon
    self.num_samples =self.config.planner.num_samples
    self.mixture_coef = self.config.planner.mixture_coef
    self.num_elites = self.config.planner.num_elites
    self.temperature = self.config.planner.temperature
    self.iterations = self.config.planner.iterations
    self.momentum = self.config.planner.momentum
    self.init_std = self.config.planner.init_std
    self.cost_limit = self.config.cost_limit

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def cost_from_recon(self, recon):
    hazards_size = 0.25
    batch_size = recon.shape[0] * recon.shape[1]
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    condition = jnp.less_equal(hazards_dist, hazards_size)
    cost = jnp.where(condition, 1.0, 0.0)
    cost = cost.sum(1)
    condition = jnp.greater_equal(cost, 1.0)
    cost = jnp.where(condition, 1.0, 0.0)


    cost = cost.reshape(recon.shape[0], recon.shape[1])
    return cost

  def true_function(self, traj_ret, traj_cost, traj_pi_gaus):
    return -traj_cost, traj_pi_gaus['action']

  def false_function(self, traj_ret, traj_cost, traj_pi_gaus):
    mask = traj_cost < self.cost_limit
    safe_elite_idxs = jnp.nonzero(lax.convert_element_type(mask, jnp.int32), size=mask.size)[0]
    return traj_ret[safe_elite_idxs], traj_pi_gaus['action'][:,safe_elite_idxs,:]

  def func_with_if_else(self, num_safe_traj, ret, cost, traj_pi_gaus):
    elite_value, elite_actions = lax.cond(num_safe_traj < self.num_elites,lambda ret, cost, traj_pi_gaus: self.true_function(ret, cost, traj_pi_gaus), lambda ret, cost, traj_pi_gaus: self.false_function(ret, cost, traj_pi_gaus), ret, cost, traj_pi_gaus)
    return elite_value, elite_actions

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """

    num_pi_trajs = int(self.mixture_coef * self.num_samples)
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v,num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, self.horizon)

    std = self.init_std * jnp.ones((self.horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      mean = jnp.zeros((self.horizon+1,)+self.act_space.shape)

    for i in range(self.iterations):
      latent_gaus = {k: jnp.repeat(v,self.num_samples,0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(current_mean + current_std * jax.random.normal(nj.rng(),(self.num_samples,)+self.act_space.shape),-1, 1)
      traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)
      else:
        traj_pi_gaus = traj_gaus
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      recon = self.wm.heads['decoder'](traj_pi_gaus)

      if self.config.use_cost:
        cost, cost_ret, cost_value = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        short_cost = cost.sum(0)
        traj_cost = cost.sum(0)
        traj_cost = traj_cost * (1000/ self.horizon)
        traj_ret_penalty = ret[0] - state['lagrange_penalty'] * cost_ret[0]
      else:
        cost = self.cost_from_recon(recon['observation'].mode())
        # [horizon,num_samples]
        traj_cost = cost.sum(0) * (1000/ self.horizon)
        # [num_samples]

      num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost<self.cost_limit, jnp.int32))


      elite_value, elite_actions = self.func_with_if_else(num_safe_traj, traj_ret_penalty, traj_cost, traj_pi_gaus)


      elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
      elite_value, elite_actions = elite_value[elite_idxs], elite_actions[:,elite_idxs,:]
      _mean = elite_actions.mean(axis=1)
      _std = elite_actions.std(axis=1)
      mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

    a = mean[0]
    return {'action': jnp.expand_dims(a,0), 'log_action_mean': jnp.expand_dims(mean.mean(axis=0),0), 'log_action_std': jnp.expand_dims(std.mean(axis=0),0), 'log_plan_num_safe_traj': jnp.expand_dims(num_safe_traj,0), 'log_plan_ret': jnp.expand_dims(ret[0,:].mean(axis=0),0), 'log_plan_cost':jnp.expand_dims(traj_cost.mean(axis=0),0)}, {'action_mean': mean, 'action_std': std}

class CCEPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.config = config
    self.ac = ac
    self.wm = wm
    self.act_space = act_space

    self.horizon = self.config.planner.horizon
    self.num_samples =self.config.planner.num_samples
    self.mixture_coef = self.config.planner.mixture_coef
    self.num_elites = self.config.planner.num_elites
    self.iterations = self.config.planner.iterations
    self.momentum = self.config.planner.momentum
    self.init_std = self.config.planner.init_std
    self.cost_limit = self.config.cost_limit

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def cost_from_recon(self, recon):
    hazards_size = 0.25
    batch_size = recon.shape[0] * recon.shape[1]
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    condition = jnp.less_equal(hazards_dist, hazards_size)
    cost = jnp.where(condition, 1.0, 0.0)
    cost = cost.sum(1)
    # condition = jnp.greater_equal(cost, 1.0)
    # cost = jnp.where(condition, 1.0, 0.0)


    cost = cost.reshape(recon.shape[0], recon.shape[1])
    return cost

  def true_function(self, ret, cost, traj_pi_gaus):
    return -cost, traj_pi_gaus['action']

  def false_function(self, ret, cost, traj_pi_gaus):
    mask = cost < self.cost_limit
    safe_elite_idxs = jnp.nonzero(lax.convert_element_type(mask, jnp.int32), size=mask.size)[0]
    return ret[safe_elite_idxs], traj_pi_gaus['action'][:,safe_elite_idxs,:]

  def func_with_if_else(self, num_safe_traj, ret, cost, traj_pi_gaus):
    elite_value, elite_actions = lax.cond(num_safe_traj < self.num_elites,lambda ret, cost, traj_pi_gaus: self.true_function(ret, cost, traj_pi_gaus), lambda ret, cost, traj_pi_gaus: self.false_function(ret, cost, traj_pi_gaus), ret, cost, traj_pi_gaus)
    return elite_value, elite_actions

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """

    num_pi_trajs = int(self.mixture_coef * self.num_samples)
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v,num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, self.horizon)
    std = self.init_std * jnp.ones((self.horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      mean = jnp.zeros((self.horizon+1,)+self.act_space.shape)

    for i in range(self.iterations):
      latent_gaus = {k: jnp.repeat(v,self.num_samples,0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        return jnp.clip(current_mean + current_std * jax.random.normal(nj.rng(),(self.num_samples,)+self.act_space.shape),-1, 1)
      traj_gaus = self.wm.imagine(policy, latent_gaus, self.horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)
      else:
        traj_pi_gaus = traj_gaus
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)
      traj_rew = ret[0]
      if self.config.use_cost:
        cost, cost_ret, cost_value = self.ac.cost_critics['extr'].score(traj_pi_gaus)
        # [horizon,num_samples]
        traj_cost = cost.sum(0)
        # [num_samples]
        traj_cost = traj_cost * (1000/ self.horizon)

      else:
        recon = self.wm.heads['decoder'](traj_pi_gaus)
        cost = self.cost_from_recon(recon['observation'].mode())
        # [horizon,num_samples]
        traj_cost = cost.sum(0)
        # [num_samples]

      num_safe_traj = jnp.sum(lax.convert_element_type(traj_cost<self.cost_limit, jnp.int32))

      elite_value, elite_actions = self.func_with_if_else(num_safe_traj, traj_rew, traj_cost, traj_pi_gaus)

      elite_idxs = jax.lax.top_k(elite_value, self.num_elites)[1]
      elite_value, elite_actions = elite_value[elite_idxs], elite_actions[:,elite_idxs,:]
      _mean = elite_actions.mean(axis=1)
      _std = elite_actions.std(axis=1)
      mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

    a = mean[0]
    return {'action': jnp.expand_dims(a,0), 'log_action_mean': jnp.expand_dims(mean.mean(axis=0),0), 'log_action_std': jnp.expand_dims(std.mean(axis=0),0), 'log_plan_num_safe_traj': jnp.expand_dims(num_safe_traj,0), 'log_plan_ret': jnp.expand_dims(ret[0,:].mean(axis=0),0), 'log_plan_cost':jnp.expand_dims(traj_cost.mean(axis=0),0)}, {'action_mean': mean, 'action_std': std}


class CEMPlanner(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.ac = ac
    self.wm = wm
    self.act_space = act_space
    self.config = config
  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """
    horizon = self.config.planner.horizon
    num_samples =self.config.planner.num_samples
    mixture_coef = self.config.planner.mixture_coef
    num_elites = self.config.planner.num_elites
    temperature = self.config.planner.temperature
    iterations = self.config.planner.iterations
    momentum = self.config.planner.momentum
    init_std = self.config.planner.init_std
    cost_limit = self.config.cost_limit

    num_pi_trajs = int(mixture_coef * num_samples)
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v,num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros(latent_pi['deter'].shape[0])
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, horizon)

    std = init_std * jnp.ones((horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=0)
      mean = mean.at[-1].set(mean[-2])
    else:
      mean = jnp.zeros((horizon+1,)+self.act_space.shape)

    for i in range(iterations):
      latent_gaus = {k: jnp.repeat(v,num_samples,0) for k, v in latent.items()}
      latent_gaus['is_terminal'] = jnp.zeros(latent_gaus['deter'].shape[0])
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][horizon]
        current_std = s['action_std'][horizon]
        action = jnp.clip(current_mean + current_std * jax.random.normal(nj.rng(),(num_samples,)+self.act_space.shape),-1, 1)
        return action
      traj_gaus = self.wm.imagine(policy, latent_gaus, horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)
      else:
        traj_pi_gaus = traj_gaus
      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      # [horizon,num_samples]
      traj_rew = rew.sum(0)

      elite_idxs = jax.lax.top_k(traj_rew,num_elites)[1]
      elite_value, elite_actions = traj_rew[elite_idxs], traj_pi_gaus['action'][:,elite_idxs,:]
      # [num_samples] traj:horizon,num_elites,dim]

      # Update parameters
      max_value = elite_value.max()
      score = jnp.exp(temperature*(elite_value - max_value))
      score /= score.sum(0)
      _mean = jnp.sum(jnp.expand_dims(score,1) * elite_actions , axis=1) / (score.sum(0) + 1e-9) 
      #[num_elites,1] * [horizon, num_elites,action_dim] -> [horizon,action_dim]

      _std = jnp.sqrt(jnp.sum(jnp.expand_dims(score,1) * (elite_actions -jnp.expand_dims(_mean,1)) ** 2, axis=1) / (score.sum(0) + 1e-9))
      mean, std = momentum * mean + (1 - momentum) * _mean, _std


    a = mean[0]
    return {'action': jnp.expand_dims(a,0), 'log_action_mean': jnp.expand_dims(mean.mean(axis=0),0), 'log_action_std': jnp.expand_dims(std.mean(axis=0),0), 'log_plan_num_safe_traj': jnp.zeros(1), 'log_plan_ret': jnp.expand_dims(ret[0,:].mean(axis=0),0), 'log_plan_cost': jnp.zeros(1)}, {'action_mean': mean, 'action_std': std}


class CEMPlanner_parallel(nj.Module):

  def __init__(self, ac, wm, act_space, config):
    self.ac = ac
    self.wm = wm
    self.act_space = act_space
    self.config = config
  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def train(self, imagine, start, data):
    pass

  def report(self, data):
    return {}

  def policy(self, latent, state):
    """
    Plan next action using TD-MPC inference.
    obs: raw input observation.
    eval_mode: uniform sampling and action noise is disabled during evaluation.
    step: current time step. determines e.g. planning horizon.
    t0: whether current step is the first step of an episode.
    """
    horizon = self.config.planner.horizon
    num_samples =self.config.planner.num_samples
    mixture_coef = self.config.planner.mixture_coef
    num_elites = self.config.planner.num_elites
    temperature = self.config.planner.temperature
    iterations = self.config.planner.iterations
    momentum = self.config.planner.momentum
    init_std = self.config.planner.init_std
    cost_limit = self.config.cost_limit

    num_pi_trajs = int(mixture_coef * num_samples)
    env_amount = latent['deter'].shape[0]
    latent = {k: jnp.expand_dims(v,0) for k, v in latent.items()}
    if num_pi_trajs > 0:
      latent_pi = {k: jnp.repeat(v, num_pi_trajs,0) for k, v in latent.items()}
      latent_pi['is_terminal'] = jnp.zeros((latent_pi['deter'].shape[0], latent_pi['deter'].shape[1]))
      policy = lambda s: self.ac.actor(sg(s)).sample(seed=nj.rng())
      traj_pi = self.wm.imagine(policy, latent_pi, horizon)

    std = init_std * jnp.ones((env_amount, horizon+1,)+self.act_space.shape)
    if 'action_mean' in state.keys():
      mean = jnp.roll(state['action_mean'], -1, axis=1)
      mean = mean.at[:, -1].set(mean[:, -2])
    else:
      mean = jnp.zeros((env_amount, horizon+1,)+self.act_space.shape)


    for i in range(iterations):
      latent_gaus = {k: jnp.repeat(v,num_samples,0) for k, v in latent.items()} 
      latent_gaus['is_terminal'] = jnp.zeros((latent_gaus['deter'].shape[0], latent_gaus['deter'].shape[1]))
      latent_gaus['action_mean'] = mean
      latent_gaus['action_std'] = std
      def policy(s,horizon):
        current_mean = s['action_mean'][:, horizon]
        current_std = s['action_std'][:, horizon]
        noise = jax.random.normal(nj.rng(),(num_samples, env_amount,) + self.act_space.shape)
        action = jnp.clip(jnp.expand_dims(current_mean,0)+ jnp.expand_dims(current_std,0) * noise, -1, 1)
        return action
      traj_gaus = self.wm.imagine(policy, latent_gaus, horizon, use_planner=True)

      if num_pi_trajs > 0:
        traj_pi_gaus ={}
        for k, v in traj_pi.items():
          traj_pi_gaus[k] = jnp.concatenate([traj_pi[k],traj_gaus[k]],axis=1)


      elif num_pi_trajs == 0:
        traj_pi_gaus = traj_gaus
      else:
        raise NotImplementedError

      rew, ret, value = self.ac.critics['extr'].score(traj_pi_gaus)

      ret_weight = ret * sg(traj_pi_gaus['weight'])[:-1]



      sum_rew = rew.sum(0)

      traj_reward = jnp.transpose(ret_weight[0,:,:]) # [num_samples, env.amount] to [env.amount, num_samples]
      elite_idxs = jax.lax.top_k(traj_reward, num_elites)[1] # [env.amount, k]
      elite_value = jnp.zeros((env_amount, num_elites, 1))
      elite_actions = jnp.zeros((env_amount, horizon+1, num_elites, self.act_space.shape[0]))
      for i in range(env_amount):
        elite_value = elite_value.at[i].set(jnp.expand_dims(traj_reward[i, elite_idxs[i]],axis=1))
        env_actions = traj_pi_gaus['action'][:, :, i, :] # [horizon+1, num_gaus+num_pim, action_dim]
        env_actions = env_actions[:, elite_idxs[i], :]
        elite_actions = elite_actions.at[i].set(env_actions) #[amount, k]
      # Update parameters
      max_value = elite_value.max(axis=1)
      score = jnp.exp(temperature*(elite_value - jnp.expand_dims(max_value,axis=1)))
      score /= jnp.expand_dims(score.sum(axis=1), axis=1) # [amount, k], elite_actions: [horizon,amount,k,dim]
      _mean = jnp.sum(jnp.expand_dims(score, axis=1) * elite_actions , axis=2) / jnp.expand_dims((score.sum(axis=1) + 1e-9), axis=1) # [horizon,amount,dim]
      _std = jnp.sqrt(jnp.sum(jnp.expand_dims(score, axis=1) * (elite_actions - jnp.expand_dims(_mean,2)) ** 2, axis=2) / jnp.expand_dims(score.sum(axis=1) + 1e-9, axis=1))

      mean, std = momentum * mean + (1 - momentum) * _mean, _std
      # (env_amount, num_elites, 1)

    a = mean[:,0,:]
    return {'action': a, 'log_plan_action_mean': mean.mean(axis=1), 'log_plan_action_std': std.mean(axis=1), 'log_plan_num_safe_traj': jnp.zeros(env_amount), 'log_plan_ret': traj_reward[:,:].mean(axis=1), 'log_plan_cost': jnp.zeros(env_amount)}, {'action_mean': mean, 'action_std': std} #{'action_mean': mean, 'action_std': std, 'plan_num_safe_traj': jnp.zeros(env_amount), 'plan_cost':jnp.zeros(env_amount), 'plan_ret': traj_reward[:,:].mean(axis=1)}
