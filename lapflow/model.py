"""Scale-specific transformer, coarse-first indexing, no dense causal mask.

Each scale queries itself and coarser scales. Chunked queries are mathematically
identical to full attention (up to floating point rounding). Checkpointing
recomputes blocks during backward. No recurrent physical-time forecasting.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def position3d(side, width, device):
    k = torch.arange(width//6, device=device).float()
    frequencies = 10000**(-k/max(width//6, 1))
    axes = [torch.arange(side, device=device)]*3
    coordinates = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    phase = coordinates.float()[..., None]*frequencies
    return torch.cat([phase.sin(), phase.cos()], -1).reshape(side**3, width)


def causal_attention(queries, keys, values, chunk, scale=None):
    outputs = []
    for i, query in enumerate(queries):
        key, value = torch.cat(keys[:i+1], -2), torch.cat(values[:i+1], -2)
        parts = [F.scaled_dot_product_attention(query[..., j:j+chunk, :], key, value,
                                                dropout_p=0., scale=scale)
                 for j in range(0, query.shape[-2], chunk)]
        outputs.append(torch.cat(parts, -2))
    return outputs


class ScaleBlock(nn.Module):
    def __init__(self, width, heads, levels, query_chunk):
        super().__init__()
        self.heads, self.chunk, self.width = heads, query_chunk, width
        self.qkv = nn.ModuleList([nn.Linear(width, 3*width) for _ in range(levels)])
        self.proj = nn.ModuleList([nn.Linear(width, width) for _ in range(levels)])
        self.mod = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width)) for _ in range(levels)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(width, 4*width), nn.GELU(approximate='tanh'),
                                               nn.Linear(4*width, width)) for _ in range(levels)])
        for module in self.mod:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, states, condition):
        queries, keys, values, mods = [], [], [], []
        for i, x in enumerate(states):
            shift, scale, gate, shift2, scale2, gate2 = self.mod[i](condition).unsqueeze(1).chunk(6, -1)
            mods.append((gate, shift2, scale2, gate2))
            h = F.layer_norm(x, (self.width,))*(1+scale)+shift
            qkv = self.qkv[i](h).reshape(x.shape[0], x.shape[1], 3, self.heads, self.width//self.heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4)
            queries.append(query)
            keys.append(key)
            values.append(value)
        attention = causal_attention(queries, keys, values, self.chunk,
                                     scale=.25/math.sqrt(self.width//self.heads))
        outputs = []
        for i, (x, attended) in enumerate(zip(states, attention)):
            gate, shift, scale, gate2 = mods[i]
            x = x+gate*self.proj[i](attended.transpose(1, 2).reshape_as(x))
            x = x+gate2*self.mlp[i](F.layer_norm(x, (self.width,))*(1+scale)+shift)
            outputs.append(x)
        return tuple(outputs)


class LapFlow3D(nn.Module):
    def __init__(self, latent_channels=8, width=192, depth=6, heads=3, patch=2,
                 levels=3, query_chunk=256, checkpoint=True):
        super().__init__()
        self.patch, self.width, self.levels = patch, width, levels
        self.checkpoint, self.channels = checkpoint, latent_channels
        self.embed = nn.ModuleList([nn.Conv3d(latent_channels, width, patch, stride=patch) for _ in range(levels)])
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([ScaleBlock(width, heads, levels, query_chunk) for _ in range(depth)])
        self.final_mod = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width)) for _ in range(levels)])
        self.head = nn.ModuleList([nn.Linear(width, latent_channels*patch**3) for _ in range(levels)])
        for head in self.head:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, states, t):
        if not 1 <= len(states) <= self.levels:
            raise ValueError('Invalid active scale count')
        if t.ndim == 0:
            t = t.expand(states[0].shape[0])
        frequency = torch.exp(-math.log(10000)*torch.arange(32, device=t.device)/32)
        phase = t.float()[:, None]*1000*frequency
        condition = self.time(torch.cat([phase.cos(), phase.sin()], -1))
        tokens = []
        for i, x in enumerate(states):
            if len(set(x.shape[-3:])) != 1 or x.shape[-1] % self.patch:
                raise ValueError('Expected patch-divisible cubic latent')
            h = self.embed[i](x).flatten(2).transpose(1, 2)
            tokens.append(h+position3d(x.shape[-1]//self.patch, self.width, x.device).to(h.dtype))
        tokens = tuple(tokens)
        for block in self.blocks:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, condition, use_reentrant=False)
            else:
                tokens = block(tokens, condition)
        outputs, patch = [], self.patch
        for i, h in enumerate(tokens):
            shift, scale = self.final_mod[i](condition).unsqueeze(1).chunk(2, -1)
            h = self.head[i](F.layer_norm(h, (self.width,))*(1+scale)+shift)
            side = states[i].shape[-1]//patch
            h = h.reshape(h.shape[0], side, side, side, self.channels, patch, patch, patch)
            outputs.append(h.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
                h.shape[0], self.channels, side*patch, side*patch, side*patch))
        return tuple(outputs)
