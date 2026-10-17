"""Local 3D autoencoder, deliberately without spatial normalization/attention.

The same joint latent volume is decoded in overlapping tiles with periodic
halos. Tile cores are retained; halo predictions are discarded, not averaged.
"""
from itertools import product
import torch
from torch import nn


class Autoencoder3D(nn.Module):
    def __init__(self, widths=(8, 16, 24, 32), latent_channels=8):
        super().__init__()
        self.factor = 2**len(widths)
        self.latent_channels = latent_channels
        encoder, previous = [], 3
        for width in widths:
            encoder.extend([nn.Conv3d(previous, width, 3, stride=2, padding=1), nn.SiLU()])
            previous = width
        encoder.append(nn.Conv3d(previous, latent_channels, 1))
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv3d(latent_channels, widths[-1], 1), nn.SiLU()]
        previous = widths[-1]
        outputs = list(widths[-2::-1])+[3]
        for index, width in enumerate(outputs):
            decoder.append(nn.ConvTranspose3d(previous, width, 4, stride=2, padding=1))
            if index != len(outputs)-1:
                decoder.append(nn.SiLU())
            previous = width
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def periodic_window(x, origin, size):
    """Extract periodic [B,C,S,S,S] without padding the entire DNS volume."""
    for axis, start in enumerate(origin, 2):
        indices = torch.arange(start, start+size, device=x.device) % x.shape[axis]
        x = x.index_select(axis, indices)
    return x


def core(x, halo, size):
    return x[..., halo:halo+size, halo:halo+size, halo:halo+size]


def check_tiling(model, grid, tile, halo):
    factor = model.factor
    if grid % tile or tile % factor or halo % factor or halo < 2*factor:
        raise ValueError('Invalid aligned tile/halo geometry')


@torch.no_grad()
def encode_tiled(model, read_window, grid, tile, halo, device):
    """read_window(origin,size) returns normalized CPU [1,3,S,S,S]."""
    check_tiling(model, grid, tile, halo)
    factor = model.factor
    output = torch.empty(1, model.latent_channels, *([grid//factor]*3))
    for origin in product(range(0, grid, tile), repeat=3):
        x = read_window(tuple(i-halo for i in origin), tile+2*halo).to(device)
        latent = core(model.encoder(x), halo//factor, tile//factor).float().cpu()
        slices = tuple(slice(i//factor, (i+tile)//factor) for i in origin)
        output[(...,)+slices] = latent
    return output


@torch.no_grad()
def decode_tiled(model, latent, tile, halo, device):
    """Full output stays on CPU. GPU holds one tile and its decoder features."""
    factor = model.factor
    latent = latent.cpu()
    grid = latent.shape[-1]*factor
    check_tiling(model, grid, tile, halo)
    output = torch.empty(latent.shape[0], 3, grid, grid, grid)
    for origin in product(range(0, grid, tile), repeat=3):
        start = tuple((i-halo)//factor for i in origin)
        local = periodic_window(latent, start, (tile+2*halo)//factor).to(device)
        decoded = core(model.decoder(local), halo, tile).float().cpu()
        slices = tuple(slice(i, i+tile) for i in origin)
        output[(...,)+slices] = decoded
    return output
