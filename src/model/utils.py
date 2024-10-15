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
