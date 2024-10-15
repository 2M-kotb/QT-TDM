from dataclasses import dataclass
from typing import Any, Optional, Tuple
from einops import rearrange, repeat
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from .utils import init_weights
from .gpt_transformer import Transformer 
from .slicer import Head


@dataclass
class TDMOutput:
    output_sequence: torch.FloatTensor 
    obs_predictions: torch.FloatTensor # next observation
    reward_predictions: torch.FloatTensor # reward


class TDM(nn.Module):
    def __init__(self, cfg) -> None:

        super().__init__()
        self.cfg = cfg

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        # action embeddings table
        self.action_bin_embeddings = nn.Parameter(torch.zeros(cfg.action_tokens, cfg.action_bins, cfg.embed_dim))
        
        # obs embeddings (a linear layer)
        self.obs_embeddings = nn.Linear(cfg.obs_shape[0],cfg.embed_dim)
        
        # positional embeddings
        self.pos_embeddings = nn.Embedding(cfg.max_tokens,cfg.embed_dim)

        self.ln = nn.LayerNorm(cfg.embed_dim)
        
        # GPT-like Transformer
        self.transformer = Transformer(cfg)

        # output heads (2 MLPs; one for reward, one for next obs)
        self.reward_head = Head(
            max_blocks=cfg.max_blocks,
            block_mask=self.action_token_pattern,
            head_module=nn.Sequential(
                nn.Linear(cfg.embed_dim, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 1) 
            )
        )

        self.obs_head = Head(
            max_blocks=cfg.max_blocks,
            block_mask=self.action_token_pattern,
            head_module=nn.Sequential(
                nn.Linear(cfg.embed_dim, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, cfg.obs_shape[0]) 
            )
        )

        # weights initialization
        self.apply(init_weights)
        self._initialize_last_layer()

        
    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""

        self.cfg.tokens_per_block = self.cfg.action_tokens + self.cfg.obs_tokens # N action tokens + one token for obs
        self.cfg.max_tokens = self.cfg.tokens_per_block * self.cfg.max_blocks


    def _initialize_patterns(self) -> None:
        """Initialize patterns for block masks."""
        self.obs_token_pattern = torch.zeros(self.cfg.tokens_per_block)
        self.obs_token_pattern[0] = 1
        self.action_token_pattern = torch.zeros(self.cfg.tokens_per_block)
        self.action_token_pattern[-1] = 1

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            module_to_initialize = [self.reward_head , self.obs_head]
            for head in module_to_initialize:
                for layer in reversed(head.head_module): #
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break
    
    def __repr__(self) -> str:
        return "TDM"  

    def forward(self, inp_sequence, past_keys_values=None):

        num_steps = inp_sequence.size(1)  
        assert num_steps <= self.cfg.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # add positional embeddings
        sequences = inp_sequence + self.pos_embeddings(prev_steps + torch.arange(num_steps, device=inp_sequence.device))
        # layernorm
        sequences = self.ln(sequences)
        # pass the sequence through the transformer and get the final outputs which of size [B,Lx(N+1),D]
        x = self.transformer(sequences, past_keys_values)
        # prediction heads
        obs_preds = self.obs_head(x, num_steps=num_steps, prev_steps=prev_steps)
        reward_preds = self.reward_head(x, num_steps=num_steps, prev_steps=prev_steps)

        return TDMOutput(x, obs_preds, reward_preds)


    def compute_loss(self, batch):

        observations = batch['observations'] # [B,L,obs_shape]
        actions = batch['actions'] # [B,L,N]
        batch_, seq_len,  num_actions = actions.shape
        
        # state embeddings
        state_embeddings = self.obs_embeddings(observations) # [B,L,D]
        # action embeddings
        action_embeddings = self.action_bin_embeddings[:num_actions]
        action_embeddings = repeat(action_embeddings, 'n a d -> b l n a d', b = batch_, l = seq_len)
        past_action_bins = repeat(actions, 'b l n -> b l n 1 d', d = action_embeddings.shape[-1])
        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b l n 1 d -> b l n d') # [B,L,N,D]
        # concatenate obs and action
        inp_seq = torch.cat((state_embeddings, bin_embeddings.flatten(start_dim=2)), dim=2).reshape(batch_, -1, self.cfg.embed_dim) #[B,Lx(N+1),D]
        # pass through transformer
        outputs = self(inp_seq)

        # compute reward loss
        reward_targets = batch["rewards"]
        r_mean = outputs.reward_predictions.squeeze(-1) 
        r_dist = D.Normal(r_mean, torch.ones_like(r_mean))
        r_loss = -r_dist.log_prob(reward_targets).mean()

        # compute obs loss
        obs_targets = batch['observations']
        obs_loss =  F.mse_loss(outputs.obs_predictions[:,:-1], obs_targets[:,1:], reduction='mean')

        # total loss (reward loss + obs loss)
        loss = self.cfg.reward_loss_coef * r_loss + self.cfg.obs_loss_coef * obs_loss 

        return loss, r_loss, obs_loss



    @torch.no_grad()
    def reset_from_initial_observations(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        self.keys_values = None
        # generate empty keys and values
        self.keys_values = self.transformer.generate_empty_keys_values(n=obs.shape[0], max_tokens=self.cfg.max_tokens)
        # obs embedding
        obs_embeddings = self.obs_embeddings(obs).unsqueeze(1)
        # pass it through transformer
        _ = self(obs_embeddings, past_keys_values=self.keys_values)

    @torch.no_grad()
    def step(self, actions: torch.FloatTensor, t: int) -> None:
        assert self.keys_values is not None 
        assert self.keys_values.size <= self.cfg.max_tokens

        batch, num_actions = actions.shape

        # action embeddings
        action_embeddings = self.action_bin_embeddings[:num_actions]
        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])
        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        # pass actions through transformer autoregressively (i.e., dim_by_dim)
        for i in range(num_actions):
            # pass action_bin through transformer
            outputs = self(bin_embeddings[:,i].unsqueeze(1), past_keys_values=self.keys_values)

        # next_obs
        next_obs = outputs.obs_predictions
        # reward
        reward = outputs.reward_predictions

        # pass next obs through transformer
        if t+1 < self.cfg.horizon:
            #get obs embeddings
            next_obs_embeddings = self.obs_embeddings(next_obs)
            # pass it through transformer
            _ = self(next_obs_embeddings, past_keys_values=self.keys_values)

        return next_obs.squeeze(1), reward.squeeze(1)


