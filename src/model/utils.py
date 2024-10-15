import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from einops import pack, unpack, repeat, reduce, rearrange


def init_weights(module):
    if isinstance(module, (nn.Embedding, nn.Linear, nn.Parameter)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def batch_select_indices(t, indices):
    if t.ndim == 4:
        indices = repeat(indices, 'b n -> b n 1 d', d = t.shape[-1])
        selected = t.gather(-2, indices)
        return selected.squeeze(-2)
        # indices = rearrange(indices, '... -> ... 1 1')
        # indices = indices.repeat(1,1,1,t.shape[-1])
        # selected = t.gather(-2, indices)
        # return selected.squeeze(-2)

    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')


def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == 'none' or nonlinearity == 'linear':
        return nn.Identity()
    elif nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == 'elu':
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')

@torch.jit.script
def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

#=================
# schedule epsilon
#=================
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

#=================
# schedule Horizon
#=================
def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)


#========
# MLP   #
#========
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout_p, norm, pre_activation=False, post_activation=False):
        super().__init__()

        dims = (input_dim,) + tuple(hidden_dim) + (output_dim,)
        num_layers = len(dims) - 1
        act_fn = get_activation(activation) # activation func.
        has_dropout = dropout_p != 0
        has_norm = norm is not None and norm != 'none'
        if has_dropout:
            dropout = nn.Dropout(dropout_p) 
        
        layers = []
        if pre_activation:
            if has_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(act_fn)

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn)
            if has_dropout:
                layers.append(dropout)


        layers.append(nn.Linear(dims[-2], dims[-1])) # output layer
        
        if post_activation:
            if has_norm:
                layers.append(nn.LayerNorm(output_dim))
            layers.append(act_fn)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


#============
# Discretizer
#============
class Discretize(nn.Module):
    """
    Tokenize action space by discretizing into uniform bins
    """
    def __init__(self, low, high, bins):
        super().__init__()
        self.low = low
        self.high = high
        self.bins = bins
        self.grid = np.linspace(low, high, num=bins, endpoint=False)[1:]
        
    def get_bins(self,actions):
        actions = actions.detach().cpu().numpy()
        bins = np.digitize(actions, self.grid)
        bins = torch.from_numpy(bins)
    
        return bins 


#---------------
# HLGAUSS LOSS #
#---------------
class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        device = torch.device("cuda")
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace( min_value, max_value, num_bins + 1, dtype=torch.float32, device=device)
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction='mean') -> torch.Tensor:
        target = self.transform_to_probs(target)
        return F.cross_entropy(logits, target , reduction=reduction)
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:  
        # first transform target with symlog
        target = torch.clamp(symlog(target), self.min_value, self.max_value)
        cdf_evals = torch.special.erf( (self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma) )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return symexp(torch.sum(probs * centers, dim=-1))