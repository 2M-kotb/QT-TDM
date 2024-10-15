from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from logger import make_dir
from model.utils import ema, Discretize, linear_schedule
from model.Q_transformer_model import QTransformer
from model.transformer_dynamics_model import TDM



class QT_TDM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.device = torch.device(cfg.misc.device)

        # Q-Transformer
        self.q_transformer = QTransformer(cfg.qtransformer).to(self.device)
        # EMA model (Q-Traget)
        self.q_target_model = deepcopy(self.q_transformer).requires_grad_(False)
        # Transformer Dynamics Model (TDM)
        self.dynamics_model = TDM(cfg.tdm).to(self.device)

        # used to discretize the sampled actions from Gaussians during planning
        self.action_discretizer = Discretize(low=-1.0, high=1.0, bins=cfg.tdm.action_bins)

        # optimizers
        self.Q_optimizer = torch.optim.Adam(self.q_transformer.parameters(), lr=cfg.qtransformer.lr, betas=(0.9, 0.99), eps=cfg.qtransformer.eps, weight_decay=cfg.qtransformer.decay)
        self.Dynamics_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=cfg.tdm.lr, betas=(0.9, 0.99), eps=cfg.tdm.eps, weight_decay=cfg.tdm.decay)

        # learning rate decay used in metaworld 
        if cfg.env.domain == "metaworld":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.Q_optimizer, lambda steps: (1 - (steps/self.cfg.env.train_steps)))
        
        self.dynamics_model.eval()
        self.q_transformer.eval()
        self.q_target_model.eval()


    def __repr__(self) -> str:
        return "QT-TDM agent"

    def load(self):
        pass

    
    @torch.no_grad()
    def act(self, obs, step=None, eval_mode=False, t0=False):
        if self.cfg.planning.mpc:
            # mpc w/o terminal Q-value
            return self.plan(obs, step=step, eval_mode=eval_mode, t0=t0)
        elif self.cfg.planning.mpc_QT:
            # mpc w/ terminal Q-value
            return self.plan(obs, step=step, eval_mode=eval_mode, t0=t0, use_QT=True)
        else:
            # No planning, model-free Q-Transformer
            return self.q_transformer.get_action(obs, step=step, eval_mode=eval_mode)


    @torch.no_grad()
    def estimate_value(self, actions):
        G, discount = 0, 1
        for t in range(self.cfg.planning.horizon):
            obs, reward = self.dynamics_model.step(actions[t],t)
            G +=  discount * reward 
            discount *= self.cfg.env.discount
        
        if self.cfg.planning.mpc_QT:
            # get optimal action on the terminal obs
            action = self.q_transformer.get_optimal_actions(obs)
            # get Q_value
            terminal_Q = self.q_transformer.get_q_value(obs, action)
            # max over bins
            terminal_Q = terminal_Q.max(dim=-1).values
            # add terminal_Q of last act dim
            G += discount * terminal_Q[:, -1].unsqueeze(1) 
            # G += discount * terminal_Q.mean(dim=1,keepdim=True)
        return G


    @torch.no_grad()    
    def plan(self, obs, step=None, eval_mode=False, t0=False, use_QT=False):
        #generate random actions for seed_steps
        if step <= self.cfg.misc.seed_steps and not eval_mode:
            return torch.randint(0, self.cfg.tdm.action_bins, (self.cfg.env.action_dim,), device = self.device)

        # current obs
        obs = torch.FloatTensor(obs).to(self.device)
        
        # sample QTransformer trajectories
        if use_QT and self.cfg.planning.num_Q_trajs > 0:
            Q_actions = torch.empty(self.cfg.planning.horizon, self.cfg.planning.num_Q_trajs, self.cfg.env.action_dim, dtype=torch.long, device=self.device)
            obs_ = obs.repeat(self.cfg.planning.num_Q_trajs,1)
            # reset the dynamics model with the init obs
            self.dynamics_model.reset_from_initial_observations(obs_)
            for t in range(self.cfg.planning.horizon):
                Q_actions[t] = self.q_transformer.get_optimal_actions(obs_)#, prob_random_action=0
                obs_, _ = self.dynamics_model.step(Q_actions[t], t)



        #initialize mean and std of normal distributions
        mean = torch.zeros(self.cfg.planning.horizon, self.cfg.env.action_dim, device=self.device)
        std = 2*torch.ones(self.cfg.planning.horizon, self.cfg.env.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]


        # Iterate CEM
        for i in range(self.cfg.planning.iterations):

            # sample actions--> [Horizon, num_samples, dim(A)]
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.planning.horizon, self.cfg.planning.num_samples, self.cfg.env.action_dim, device=std.device), -1, 1)

            # discretize sampled continuous actions
            actions = self.action_discretizer.get_bins(actions).to(self.device)

            if use_QT and self.cfg.planning.num_Q_trajs > 0:
                actions = torch.cat([actions, Q_actions], dim=1)


            # reset the dynamics model with the init obs
            obs_ = obs.repeat(actions.shape[1],1)
            self.dynamics_model.reset_from_initial_observations(obs_)

            # compute elite actions
            value = self.estimate_value(actions)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.planning.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # transform elite_actions from discretize_bins to continuous to update mu and sigma
            cont_elite_actions = (elite_actions/self.cfg.tdm.action_bins*2-1).clamp_(-1,1) 

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.planning.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * cont_elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (cont_elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(0.05, 2)

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]

        if not eval_mode:
            # add some noise
            prob_random_action = 0.1
            random_mask = torch.zeros_like(a).float().uniform_(0., 1.) < prob_random_action
            random_actions = torch.randint(0, self.cfg.tdm.action_bins, (self.cfg.env.action_dim,), device = self.device)
            a = torch.where(
                    random_mask,
                    random_actions,
                    a
                )
        
        return a


    def update(self, buffer, step):

        metrics = {}
        self.dynamics_model.train()
        self.q_transformer.train()
        self.q_target_model.train()
        self.Q_optimizer.zero_grad()
        self.Dynamics_optimizer.zero_grad()

        # sample batch
        batch = buffer.sample_batch(self.cfg.tdm.batch_size, self.cfg.tdm.max_blocks)
        batch = self._to_device(batch)


        # Dynamics model update
        loss, r_loss, obs_loss = self.dynamics_model.compute_loss(batch)
        loss.backward()
        if self.cfg.tdm.grad_clip is not None:
            dyn_grad = nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.cfg.tdm.grad_clip)
        self.Dynamics_optimizer.step()

        
        # Don't update Q-Transformer in case of using MPC without terminal Q-value (i.e., planning w/o QT)
        if self.cfg.planning.mpc_QT:  
            # QTransformer update
            q_loss, td_loss, conserv_loss = self.q_transformer.compute_loss(batch, self.q_target_model)
            q_loss.backward()
            if self.cfg.qtransformer.grad_clip is not None:
                Q_grad = nn.utils.clip_grad_norm_(self.q_transformer.parameters(), self.cfg.qtransformer.grad_clip) 
            self.Q_optimizer.step()

            # decay q_transformer lr in metaworld tasks
            if self.cfg.env.domain == "metaworld":
                self.lr_scheduler.step()
            lr = self.Q_optimizer.param_groups[0]["lr"]

            # update target-q network
            if step % self.cfg.qtransformer.updtae_freq==0:
                ema(self.q_transformer, self.q_target_model, self.cfg.qtransformer.tau)


        self.dynamics_model.eval()
        self.q_transformer.eval()
        self.q_target_model.eval()

        if self.cfg.planning.mpc_QT:
            metrics = {"reward_loss": r_loss.item(), "obs_loss": obs_loss.item(), "dyn_loss": loss.item(), "dyn_grad":  dyn_grad.item(),
                "q_loss":q_loss.item(), "td_loss":td_loss.item(),"conservative_loss":conserv_loss.item(), "q_grad":Q_grad.item(), "q_lr":lr}
        else:
            metrics = {"reward_loss": r_loss.item(), "obs_loss": obs_loss.item(), "dyn_loss": loss.item(), "dyn_grad":  dyn_grad.item()}

        return metrics


    

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}


        



