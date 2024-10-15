"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache



class Transformer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_tokens = cfg.max_tokens 
        self.drop = nn.Dropout(cfg.embed_pdrop)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.cfg.num_heads, max_tokens, self.cfg.embed_dim, self.cfg.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    '''implement a transformer block (i.e., Masaked self-attention and Feed-Forward Network'''
    def __init__(self, cfg) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim) # layerNorm of self-attention
        self.ln2 = nn.LayerNorm(cfg.embed_dim) # layerNorm of FFNN
        self.attn = SelfAttention(cfg) # masked self-attention
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values) # masked self-attention layer
        x = x + x_attn 
        x = x + self.mlp(self.ln2(x)) # FFNN layer
        return x


class SelfAttention(nn.Module):
    '''implements Masked self-attention layer'''
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        assert cfg.attention in ('causal', 'block_causal')
        self.num_heads = cfg.num_heads
        self.key = nn.Linear(cfg.embed_dim, cfg.embed_dim) # Key matrix W_k (256x256)
        self.query = nn.Linear(cfg.embed_dim, cfg.embed_dim) # Query matrix W_q (256x256)
        self.value = nn.Linear(cfg.embed_dim, cfg.embed_dim) # Value matrix W_v (256x256)
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim) # projection matrix to project the concatenating multiple attention heads vector into a nother vector that fed into FFNN 

        causal_mask = torch.tril(torch.ones(cfg.max_tokens, cfg.max_tokens)) # attention mask
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(cfg.tokens_per_block, cfg.tokens_per_block) for _ in range(cfg.max_blocks)]))
        self.register_buffer('mask', causal_mask if cfg.attention == 'causal' else block_causal_mask) # save attention mask as register buffer

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        '''
        x: is the input sentence with len 20 and every timestep has 17 tokens (16 frames + 1 action), so in total it is 340 tokens
        B, T, C: batch size, num_of_tokens (340), dim of each token (256)
        '''
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        # create q,k,v and reshape them w.r.t num of heads
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs) = (B, 4, 340, 64)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs) = (B, 4, 340, 64)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs) = (B, 4, 340, 64)
        
        # During planning, where a token is feed every time step
        # q,k,v for the current token are computed in the previous steps, then k and v are saved to KV_cache
        # Finally, k and v of this token plus the previous tokens are retrived from KV_cache, so the attention of the current token can be 
        # computed against all previous tokens.
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get() # k,v for the current tokens and previous tokens

        # computing self-attention
        #-------------------------

        # compute score by multiplying q and k and divide by the square root of k. The scoring matrix shape is (B, 4, 340, 340)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # perform masked self-attention by adding -inf to the future tokens
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        # compute score probability using softmax
        att = F.softmax(att, dim=-1)
        # perform dropout in the attention matrix
        att = self.attn_drop(att)
        # compute final attention z by multiplying attention matrix by value. The final attention z size is (B, 4, 340, 64)
        y = att @ v
        # concatinate the 4 heads final attention. The final attention z size will be (B, 340, 256)
        y = rearrange(y, 'b h t e -> b t (h e)')
        # perform residual dropout
        y = self.resid_drop(self.proj(y))

        return y
