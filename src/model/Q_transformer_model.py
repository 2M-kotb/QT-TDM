from einops import rearrange, pack, unpack, reduce, repeat
from dataclasses import dataclass
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from random import random
from .utils import init_weights, default, exists, LinearSchedule, batch_select_indices, zero_
from .gpt_transformer import Transformer 

class QTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # action embeddings table
        self.action_bin_embeddings = nn.Parameter(torch.zeros(self.num_actions, cfg.action_bins, cfg.embed_dim))

        # obs embeddings (a linear layer)
        self.obs_embeddings = nn.Linear(cfg.obs_shape[0],cfg.embed_dim)

        # positional embeddings
        self.pos_embeddings = nn.Embedding(cfg.max_tokens,cfg.embed_dim)

        # GPT-like Transformer
        self.transformer = Transformer(cfg) 

        # epsilon decay schedule
        self.exploration = LinearSchedule(cfg.train_steps*0.3, cfg.final_epsilon) 
        
        # 2 Q-Networks as MLPs
        self.q_head_1 = nn.Sequential(
                    nn.Linear(cfg.embed_dim, cfg.embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(cfg.embed_dim, cfg.action_bins)
                )
        
        self.q_head_2 = nn.Sequential(
                    nn.Linear(cfg.embed_dim, cfg.embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(cfg.embed_dim, cfg.action_bins)
                )

            
        # weights initialization
        self.apply(init_weights)
        zero_([self.q_head_1[-1].weight, self.q_head_2[-1].weight])

    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
        self.cfg.max_tokens = self.cfg.tokens_per_block * self.cfg.max_blocks
        self.num_actions = self.cfg.tokens_per_block

    @property
    def device(self):
        return self.action_bin_embeddings.device
    
    def __repr__(self) -> str:
        return "Q-Transformer"

    
    def forward(self, inp_sequence, past_keys_values=None):

        num_steps = inp_sequence.size(1)  
        assert num_steps <= self.cfg.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        # add positional embeddings
        sequences = inp_sequence + self.pos_embeddings(prev_steps + torch.arange(num_steps, device=inp_sequence.device))
        # pass the sequence through the transformer and get the final outputs which of size [B,L,embed_size]
        hiddens = self.transformer(sequences, past_keys_values)
        # Q-Networks
        q_values_1 = self.q_head_1(hiddens)
        q_values_2 = self.q_head_2(hiddens)

        return q_values_1, q_values_2
    
    @torch.no_grad()
    def get_action(self, obs, step=None, eval_mode=False):

        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

        if eval_mode:
            return self.get_optimal_actions(obs)

        if step <= self.cfg.seed_steps:
            return self.get_random_action(obs.shape[0]).squeeze(0)

        # epsilon greedy
        epsilon = self.exploration.value(step)
        if random() < epsilon:
            return self.get_random_action(obs.shape[0]).squeeze(0)
        else:
            return self.get_optimal_actions(obs)

        
    @torch.no_grad()
    def get_random_action(self, batch, num_actions = None):
        num_actions = default(num_actions, self.num_actions)
        return torch.randint(0, self.cfg.action_bins, (batch, num_actions), device = self.device)


    @torch.no_grad()
    def get_optimal_actions(
        self, 
        obs,
        prob_random_action: float = 0.1
    
    ):

        assert 0. <= prob_random_action <= 1.
        batch = obs.shape[0]

        if prob_random_action == 1:
            return self.get_random_action(batch)

        action_embeddings = repeat(self.action_bin_embeddings, 'n a d -> b n a d', b = batch)
        action_embeddings = rearrange(action_embeddings,'b n a d -> n b a d')

        # generate empty keys and values
        keys_values = self.transformer.generate_empty_keys_values(n=batch, max_tokens=self.cfg.max_tokens)
        # obs embeddings
        obs_embeddings = self.obs_embeddings(obs).unsqueeze(1)

        action_bins = []

        tokens = obs_embeddings

        for action_idx in range(self.num_actions):

            q_values_1, q_values_2 = self(tokens, past_keys_values=keys_values)
            q_values = (q_values_1 + q_values_2) / 2
            # select action bin with the highest q_value
            selected_action_bins = q_values.argmax(dim = -1) # [b,1]
            
            if prob_random_action > 0.:
                random_mask = torch.zeros_like(selected_action_bins).float().uniform_(0., 1.) < prob_random_action
                random_actions = self.get_random_action(batch, 1)
                #random_actions = rearrange(random_actions, '... 1 -> ...')


                selected_action_bins = torch.where(
                    random_mask,
                    random_actions,
                    selected_action_bins
                )

            # embed the selected action bin
            next_action_embed = action_embeddings[action_idx] # [b,a,d]
            selected_action_bins_ = repeat(selected_action_bins, 'b 1 -> b 1 d', d = action_embeddings.shape[-1])
            next_action_embed = next_action_embed.gather(-2, selected_action_bins_)
            
            tokens = next_action_embed

            action_bins.append(selected_action_bins)

        action_bins = torch.stack(action_bins, dim = -1).squeeze(1)


        return action_bins

    def get_q_value(self, obs, actions):

        # embed state and actions using model
        obs_embeddings = self.obs_embeddings(obs)

        batch_, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch_)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        # concatenate state and bin embeddings
        tokens, _ = pack((obs_embeddings, bin_embeddings), 'b * d')
        tokens = tokens[:, :num_actions]

        q_values_1, q_values_2 = self(tokens)
        # get the avg 
        q_values = (q_values_1 + q_values_2) / 2

        return q_values


    def compute_loss(self, batch, ema_model):


        # we only use the first nstep+1 from the sampled sequence batch
        all_states = batch['observations'][:, :self.cfg.n_step_td+1]
        all_actions = batch['actions'][:, :self.cfg.n_step_td+1]
        all_rewards = batch['rewards'][:, :self.cfg.n_step_td+1]
        all_returns = batch['return_to_go'][:, :self.cfg.n_step_td+1]

        # delete batch from Gpu to save memory
        del batch

        # obs and action at time step t: (s_t, a_t)
        states = all_states[:, 0]
        actions = all_actions[:, 0]
        MC_return = all_returns[:, 0]

        # obs and action at time step t+n (i.e., target): (s_{t+n}, a_{t+n})
        next_states = all_states[:, -1]
        next_actions = all_actions[:, -1]
        next_MC_return = all_returns[:, -1]


        # embed state and actions using model
        state_embeddings = self.obs_embeddings(states) #pred_states

        batch_, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch_)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        # concatenate state and bin embeddings
        tokens, _ = pack((state_embeddings, bin_embeddings), 'b * d')
        tokens = tokens[:, :num_actions] # last action bin not needed for the proposed q-learning

        # get q_values for all actions
        q_pred_all_actions_1, q_pred_all_actions_2 = self(tokens)
        # we only update the q_value that corresponds to the observed bin
        q_pred_1 = batch_select_indices(q_pred_all_actions_1, actions) #[batch, num_actions, num_classes]
        q_pred_2 = batch_select_indices(q_pred_all_actions_2, actions)

        #----------------
        # same step target
        #----------------
        

        # embed state and actions using ema model
        state_embeddings = ema_model.obs_embeddings(states)

        batch_, num_actions = actions.shape
        action_embeddings = ema_model.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch_)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        # concatenate state and bin embeddings
        tokens, _ = pack((state_embeddings, bin_embeddings), 'b * d')
        tokens = tokens[:, :num_actions] # last action bin not needed for the proposed q-learning

        # get q_target for all actions
        q_target_all_actions_1, q_target_all_actions_2 = ema_model(tokens)
        # get the min as target
        q_target_all_actions_min = torch.min(q_target_all_actions_1, q_target_all_actions_2)
        # max over bins
        q_target = q_target_all_actions_min.max(dim=-1).values


        if self.cfg.use_MC_return:
            q_target.clamp_(min=MC_return.unsqueeze(-1))


        #-----------------
        # Next step target
        #-----------------
        # embed next state and next actions using ema model
        next_state_embeddings = ema_model.obs_embeddings(next_states)

        batch_, num_actions = next_actions.shape
        next_action_embeddings = ema_model.action_bin_embeddings[:num_actions]

        next_action_embeddings = repeat(next_action_embeddings, 'n a d -> b n a d', b = batch_)
        past_action_bins = repeat(next_actions, 'b n -> b n 1 d', d = next_action_embeddings.shape[-1])

        next_bin_embeddings = next_action_embeddings.gather(-2, past_action_bins)
        next_bin_embeddings = rearrange(next_bin_embeddings, 'b n 1 d -> b n d')

        # concatenate state and bin embeddings
        tokens, _ = pack((next_state_embeddings, next_bin_embeddings), 'b * d')
        tokens = tokens[:, :num_actions] # last action bin not needed for the proposed q-learning

        # get next_q_target for n_step
        q_next_all_action_1, q_next_all_action_2 = ema_model(tokens)
        # get the min
        q_next_all_action_min = torch.min(q_next_all_action_1, q_next_all_action_2)
        # max over bins
        q_next = q_next_all_action_min.max(dim=-1).values



        if self.cfg.use_MC_return:
            q_next.clamp_(min=next_MC_return.unsqueeze(-1))


        # compute the loss for all actions except the last one
        q_pred_rest_actions_1, q_pred_last_action_1      = q_pred_1[:, :-1], q_pred_1[:, -1]
        q_pred_rest_actions_2, q_pred_last_action_2      = q_pred_2[:, :-1], q_pred_2[:, -1]
        q_target_first_action, q_target_rest_actions = q_target[:, 0], q_target[:, 1:]

       
        # losses_all_actions_but_last_1 = F.mse_loss(q_pred_rest_actions_1, q_target_rest_actions, reduction = 'none')
        # losses_all_actions_but_last_2 = F.mse_loss(q_pred_rest_actions_2, q_target_rest_actions, reduction = 'none')
        losses_all_actions_but_last_1 = F.smooth_l1_loss(q_pred_rest_actions_1, q_target_rest_actions, reduction = 'none')
        losses_all_actions_but_last_2 = F.smooth_l1_loss(q_pred_rest_actions_2, q_target_rest_actions, reduction = 'none')
       

        # compute the loss for last action
        q_target_last_action, discount = 0, 1
        for i in range(self.cfg.n_step_td):
            q_target_last_action +=  all_rewards[:,i] * discount  
            discount *= self.cfg.discount

        q_target_last_action += discount * q_next[:, 0]

        # losses_last_action_1 = F.mse_loss(q_pred_last_action_1, q_target_last_action, reduction = 'none')
        # losses_last_action_2 = F.mse_loss(q_pred_last_action_2, q_target_last_action, reduction = 'none')

        losses_last_action_1 = F.smooth_l1_loss(q_pred_last_action_1, q_target_last_action, reduction = 'none')
        losses_last_action_2 = F.smooth_l1_loss(q_pred_last_action_2, q_target_last_action, reduction = 'none')
       


        # flatten and average

        losses_1, _ = pack([losses_all_actions_but_last_1, losses_last_action_1], '*')
        losses_2, _ = pack([losses_all_actions_but_last_2, losses_last_action_2], '*')

        td_loss  = (losses_1.mean() + losses_2.mean()) / 2
        
        # compute conservative loss (Not used in online RL)
        
        batch = actions.shape[0]

        q_preds = q_pred_all_actions_1
        
        q_preds = rearrange(q_preds, '... a -> (...) a')

        num_action_bins = q_preds.shape[-1]
        num_non_dataset_actions = num_action_bins - 1


        actions = rearrange(actions, '... -> (...) 1')

        dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))

        q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
        q_actions_not_taken = rearrange(q_actions_not_taken, '(b t a) -> b t a', b = batch, a = num_non_dataset_actions)


        conservative_reg_loss = ((q_actions_not_taken - (0)) ** 2).sum() / (num_non_dataset_actions )

        # total loss
        loss =  self.cfg.td_loss_coef * td_loss + self.cfg.conservative_loss_coef * conservative_reg_loss * self.cfg.conservative_reg_loss_weight
        # conservative_reg_loss = torch.tensor(0)


        return loss, td_loss, conservative_reg_loss


