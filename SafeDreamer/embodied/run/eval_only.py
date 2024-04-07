import re

import embodied
import numpy as np
import os
from matplotlib import pylab # type: ignore
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np

def save_video(frames, video_folder, name_prefix, fps=20):
  os.makedirs(video_folder, exist_ok=True)
  video_path = os.path.join(video_folder, f"{name_prefix}.mp4")
  
  frame_height, frame_width = frames[0].shape[:2]
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
  
  for frame in frames:
      frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      out.write(frame_bgr)
  
  out.release()
  
  print(f"The video is saved at {video_path}")


class video_buffer:
  def __init__(self) -> None:
    self.groundtruth_video_list = []
    self.groundtruth_video_list2 = []
    self.groundtruth_video_far_list = []
    self.video_list_pred = []
    self.video_list_pred2 = []
    self.video_list_pred_truth = []
    self.store_time = 0

  def draw_picture(self, logdir, current_step, ground, pred):
    for i in range(0,min(15, min(len(ground), len(pred))),1):
      # cv2.imwrite('true'+str(i)+'.png', self.groundtruth_video_list[i])
      # cv2.imwrite('pred'+str(i)+'.png', self.video_list_pred[i])
      # plt.imsave(logdir+'/true' + '_' + str(self.store_time)+ '_' + str(i)+'.png', self.groundtruth_video_list[i])
      # plt.imsave(logdir+'/pred' + '_' + str(self.store_time)+ '_' + str(i)+'.png', self.video_list_pred[i])
      os.makedirs(logdir+'/'+str(current_step)+'/', exist_ok=True)
      plt.imsave(logdir+'/'+str(current_step)+'/true' + '_' + str(i)+'.png', ground[i])
      plt.imsave(logdir+'/'+str(current_step)+'/pred' + '_' + str(i)+'.png', pred[i])
def eval_only(agent, env, logger, args, lag):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  log_num = 0
  def draw_picture(
      timestep: int,
      num_episode: int,
      pred_state: np.ndarray,
      true_state: np.ndarray,
      save_replay_path: str = './',
      name: str = 'reward',
  ) -> None:
      """Draw a curve of the predicted value and the ground true value.

      Args:
          timestep (int): current step.
          num_episode (int): number of episodes.
          pred_state (list): predicted state.
          true_state (list): true state.
          save_replay_path (str): save replay path.
          name (str): name of the curve.
      """
      target1 = list(pred_state)
      target2 = list(true_state)
      input1 = np.arange(0, np.array(pred_state).shape[0], 1)
      input2 = np.arange(0, np.array(pred_state).shape[0], 1)

      pylab.plot(input1, target1, 'r-', label='pred')
      pylab.plot(input2, target2, 'b-', label='true')
      #input_min = min(np.min(pred_state),np.min(true_state))
      #input_max = max(np.max(pred_state),np.max(true_state))

      pylab.xlabel('Step')
      pylab.ylabel(name)
      pylab.xticks(np.arange(0, np.array(pred_state).shape[0], 50))  # Set the axis numbers
      if name == 'reward':
          pylab.yticks(np.arange(0, 3, 0.2))
      else:
          pylab.yticks(np.arange(0, 1, 0.2))
      pylab.legend(
          loc=3,
          borderaxespad=2.0,
          bbox_to_anchor=(0.7, 0.7),
      )  # Sets the position of that box for what each line is
      pylab.grid()  # draw grid
      pylab.savefig(
          os.path.join(
              save_replay_path,
              str(name) + str(timestep) + '_' + str(num_episode) + '.png',
          ),
          dpi=200,
      )  # save as picture
      pylab.close()
  def per_episode(ep, video_buffer):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')

    if 'cost' in ep.keys():
      cost = float(ep['cost'].astype(np.float64).sum())
      logger.add({'cost': cost}, prefix='episode')
      print(f'Episode has {length} steps and cost {cost:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    groundtruth_video_list = []
    ep_expend = {}
    for key, value in ep.items():
      ep_expend[key] = np.expand_dims(value, 0)

    model_report = agent.report_eval(ep_expend)


    if 'image_orignal' in ep.keys():
      for i in range(ep['image_orignal'].shape[0]):
        groundtruth_video_list.append(ep['image_orignal'][i])
      save_video(
        frames=groundtruth_video_list,
        video_folder=args.logdir,
        name_prefix='groundtruth_video_list_' + str(step.value),
        fps=20,
      )
    groundtruth_video_list2 = []
    if 'image_orignal2' in ep.keys():
      for i in range(ep['image_orignal2'].shape[0]):
        groundtruth_video_list2.append(ep['image_orignal2'][i])
      save_video(
        frames=groundtruth_video_list2,
        video_folder=args.logdir,
        name_prefix='groundtruth_video_list2_'+ str(step.value),
        fps=20,
      )
    groundtruth_video_far_list = []
    if 'image_far' in ep.keys():
      for i in range(ep['image_far'].shape[0]):
        groundtruth_video_far_list.append(ep['image_far'][i])
      save_video(
        frames=groundtruth_video_far_list,
        video_folder=args.logdir,
        name_prefix='groundtruth_video_far_list_'+ str(step.value),
        fps=20,
      )

    if 'openl_image2' in model_report.keys():
      resize_picture_list2 = []
      video_list_pred2 = []
      model_video2 = np.clip(255 * model_report['openl_image2'], 0, 255).astype(np.uint8)
      for i in range(model_video2.shape[0]):
        resize_picture2 = cv2.resize(model_video2[i], (1024,1024), interpolation=cv2.INTER_AREA)
        resize_picture_list2.append(resize_picture2)
        video_list_pred2.append(resize_picture2)
      save_video(
        frames=video_list_pred2,
        video_folder=args.logdir,
        name_prefix='video_list_pred2_' + str(step.value),
        fps=20,
      )

    if 'openl_image' in model_report.keys():
      resize_picture_list = []
      video_list_pred = []
      model_video = np.clip(255 * model_report['openl_image'], 0, 255).astype(np.uint8)
      for i in range(model_video.shape[0]):
        resize_picture = cv2.resize(model_video[i], (1024,1024), interpolation=cv2.INTER_AREA)
        resize_picture_list.append(resize_picture)
        video_list_pred.append(resize_picture)
      save_video(
        frames=video_list_pred,
        video_folder=args.logdir,
        name_prefix='video_list_pred_'+ str(step.value),
        fps=20,
      )

      video_list.draw_picture(args.logdir, str(step.value), groundtruth_video_list, resize_picture_list)

      # video_list_pred_truth = []
      # for i in range(model_video.shape[0]):
      #   video_list_pred_truth.append(np.concatenate([ep['image_far'][i],ep['image_orignal'][i], resize_picture_list[i], ep['image_orignal2'][i], resize_picture_list2[i]],axis=1))
      # save_video(
      #   frames=video_list_pred_truth,
      #   video_folder=args.logdir,
      #   name_prefix='video_list_pred_truth',
      #   fps=30,
      # )

    if 'openl_observation' in model_report.keys():
      pred_state = np.array(model_report['openl_observation'][0])

      pred_state_min = np.min(np.sqrt(np.sum(np.square(pred_state[:, 9:25].reshape(-1,8,2)),axis=-1)), axis=-1)
      true_state_min = np.min(np.sqrt(np.sum(np.square(ep['observation'][:,9:25].reshape(-1,8,2)),axis=-1)), axis=-1)
      draw_picture(
                    timestep=0,
                    num_episode=0,
                    pred_state=pred_state_min,
                    true_state=true_state_min,
                    save_replay_path=args.logdir,
                    name='obs_min',
                  )

      pred_state_mean = np.mean(pred_state, axis=-1)
      true_state_mean = np.mean(ep['observation'], axis=-1)
      draw_picture(
                    timestep=0,
                    num_episode=0,
                    pred_state=pred_state_mean,
                    true_state=true_state_mean,
                    save_replay_path=args.logdir,
                    name='obs_mean',
                  )
      for i in range(0, pred_state.shape[1]):
        pred_state_idx = pred_state[:,i]
        true_state_idx = ep['observation'][:,i]
        draw_picture(
                      timestep=0,
                      num_episode=0,
                      pred_state=pred_state_idx,
                      true_state=true_state_idx,
                      save_replay_path=args.logdir,
                      name='obs_'+str(i),
                    )

      pred_cost = np.array(model_report['openl_cost'][0])
      true_cost = ep['cost'][:]
      draw_picture(
                    timestep=0,
                    num_episode=0,
                    pred_state=pred_cost,
                    true_state=true_cost,
                    save_replay_path=args.logdir,
                    name='cost',
                  )
      # comput mean of loss of the true observation and the predicted observation
      loss = 0
      for i in range(pred_state.shape[0]):
        loss += np.mean(np.square(pred_state[i] - ep['observation'][i]), axis=-1)
      print("mean of loss", loss/pred_state.shape[0])


    #metrics.add(model_report, prefix='stats')

    #metrics.add(stats, prefix='stats')

  video_list = video_buffer()
  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep, video_list))
  driver.on_step(lambda tran, _: step.increment())

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    driver(policy, steps=100, lag=lag.lagrange_penalty)
    if should_log(step):
      logger.add(metrics.result())
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  logger.write()
  #video_list.store_video(args.logdir)
  #video_list.draw_picture(args.logdir)
