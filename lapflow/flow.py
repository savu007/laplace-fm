"""Linear multiscale flow matching and progressive Heun sampling."""
import torch
from torch.nn import functional as F


def pyramid(x, levels=3):
    fine = []
    for _ in range(levels-1):
        low = F.avg_pool3d(x, 2)
        fine.append(x-F.interpolate(low, scale_factor=2, mode='nearest'))
        x = low
    return [x]+list(reversed(fine))


def reconstruct(states):
    x = states[0]
    for residual in states[1:]:
        x = F.interpolate(x, scale_factor=2, mode='nearest')+residual
    return x


def noise_pyramid(noise, levels, generator=None):
    # Match the public code's distinction: two levels use Laplacian noise;
    # three use independent unit Gaussian fields. This differs from paper Alg. 1.
    if levels == 2:
        return pyramid(noise, levels)
    if levels != 3:
        raise ValueError('Only two/three-level noise supported')
    sizes = [tuple(s//factor for s in noise.shape[-3:]) for factor in (4, 2)]
    return [torch.randn((*noise.shape[:-3], *size), device=noise.device, generator=generator)
            for size in sizes]+[noise]


def path(clean, noise, t, starts, active):
    t = t.reshape(-1, 1, 1, 1, 1)
    states, targets = [], []
    for i in range(active):
        states.append((t-starts[i])/(1-starts[i])*clean[i]+(1-t)*noise[i])
        targets.append(clean[i]/(1-starts[i])-noise[i])
    return tuple(states), tuple(targets)


def loss(model, z, starts, active=None, generator=None):
    levels = len(starts)
    if active is None:
        active = int(torch.randint(1, levels+1, (), device=z.device, generator=generator))
    t = starts[active-1]+(1-starts[active-1])*torch.rand(z.shape[0], device=z.device, generator=generator)
    noise = noise_pyramid(torch.randn(z.shape, device=z.device, generator=generator), levels, generator)
    states, targets = path(pyramid(z, levels), noise, t, starts, active)
    predictions = model(states, t)
    return sum((p.float()-target.float()).square().mean() for p, target in zip(predictions, targets))


@torch.no_grad()
def sample(model, shape, starts, steps_per_stage, device, generator=None):
    """48 default Heun steps / 96 model evaluations; not physical timesteps."""
    noise = noise_pyramid(torch.randn(shape, device=device, generator=generator), len(starts), generator)
    states = []
    for stage, start in enumerate(starts):
        states.append((1-start)*noise[stage])
        end = starts[stage+1] if stage+1 < len(starts) else 1.
        dt = (end-start)/steps_per_stage
        for step in range(steps_per_stage):
            t = torch.full((shape[0],), start+step*dt, device=device)
            k1 = model(tuple(states), t)
            predicted = tuple(x+dt*k for x, k in zip(states, k1))
            k2 = model(predicted, t+dt)
            states = [x+.5*dt*(a+b) for x, a, b in zip(states, k1, k2)]
    return reconstruct(states)
