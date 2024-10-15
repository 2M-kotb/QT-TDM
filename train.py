import warnings
warnings.filterwarnings('ignore')
import os
import sys
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import argparse
import random
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from omegaconf import DictConfig
import logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from logger import make_dir
from envs import dmc_env, meta_env
from buffer import ReplayBuffer, Episode
from agent import QT_TDM 

__LOGS__ = 'logs'

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, cfg: DictConfig):

        assert torch.cuda.is_available()
        self.cfg = cfg
        set_seed(cfg.misc.seed)
        self.device = torch.device(cfg.misc.device)
        self.work_dir = Path().cwd() / __LOGS__ /cfg.env.task / cfg.env.modality / cfg.wandb.exp_name / str(cfg.misc.seed)

        # Environment
        if cfg.env.domain=='dmc_suite' or cfg.env.domain=='dmc_manip':
            self.env = dmc_env.make_env(cfg.env)
        elif cfg.env.domain=='metaworld':
            self.env = meta_env.make_env(cfg.env)
        else:
            print("Env domain should be dmc_suite or metaworld")

        # Buffer
        self.buffer = ReplayBuffer(cfg.env)
        # Agent
        self.agent = QT_TDM(self.cfg)

        print(f'{sum(p.numel() for p in self.agent.q_transformer.parameters())} parameters in agent.Q_transformer')
        print(f'{sum(p.numel() for p in self.agent.dynamics_model.parameters())} parameters in agent.dynamics')

        # # initialize agent using checkpoint
        # if cfg.path_to_checkpoint is not None:
        #     self.agent.load(cfg.path_to_checkpoint, device=self.device, load_world_model=False)

    def run(self):
        L = logger.Logger(self.work_dir, self.cfg)
        episode_idx, start_time, = 0, time.time()
        for step in range(0, self.cfg.env.train_steps+self.cfg.env.episode_length, self.cfg.env.episode_length):
            # Collect trajectory
            obs = self.env.reset()
            episode, t = Episode(self.cfg.env, obs), 0
            while not episode.done:
                action = self.agent.act(obs, step=step, t0=t==0)
                # convert actions from discretized to continuous before env step
                env_action = (action/self.cfg.qtransformer.action_bins*2-1).clamp_(-1,1)  
                obs, reward, done, info = self.env.step(env_action.cpu().numpy())
                episode += (obs, action, reward, done, info['success'])
                t += 1

            #add collected episode to buffer
            self.buffer += episode
            episode_idx += 1
            env_step = int(step*self.cfg.env.action_repeat)
            
            #update agent
            train_metrics = {}
            if step >= self.cfg.misc.seed_steps:
                num_updates =  self.cfg.env.episode_length
                for i in tqdm(range(num_updates), desc=f"Training {str(self.agent)}", file=sys.stdout):
                    train_metrics.update(self.agent.update(self.buffer, step+i))

            common_metrics = {
                'episode': episode_idx,
                'step': step,
                'env_step': env_step,
                'total_time': time.time() - start_time,
                'episode_reward': episode.cumulative_reward,
                'episode_success': episode.success,
                }

            train_metrics.update(common_metrics)
            L.log(train_metrics, category='train')

            #Evaluate agent periodically
            if env_step % self.cfg.misc.eval_freq == 0:
                # evaluate the agent
                common_metrics['episode_reward'], common_metrics['episode_success'] = self.evaluate(self.cfg.misc.eval_episodes, step, env_step, L.video)
                L.log(common_metrics, category='eval')


        L.finish(self.agent)
        print('Training completed successfully')

    def evaluate(self, num_episodes, step, env_step, video):
        """Evaluate a trained agent and optionally save a video."""
        episode_rewards, episode_success = [], []
        for i in range(num_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if video: video.init(self.env, enabled=(i==0))
            while not done:
                action = self.agent.act(obs, step=step, t0=t==0, eval_mode=True)
                # convert actions from discretized to continuous before env step
                env_action = (action/self.cfg.qtransformer.action_bins*2-1).clamp_(-1,1)  
                obs, reward, done, info = self.env.step(env_action.cpu().numpy())
                ep_reward += reward
                if video: video.record(self.env)
                t += 1
            episode_rewards.append(ep_reward)
            episode_success.append(info['success'])
            if video: video.save(env_step)
        return np.nanmean(episode_rewards), np.nanmean(episode_success)




