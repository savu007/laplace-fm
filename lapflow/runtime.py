"""Reproducibility, checkpoints, and numerical/memory diagnostics."""
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    return device


def autocast(device, enabled):
    if device.type == 'cuda' and enabled:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 requested but unsupported; set amp=false explicitly')
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, step, best, config_hash, provenance):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'best': best, 'config_hash': config_hash, 'provenance': provenance,
             'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate(), 'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == 'cuda' else None}
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, config_hash, optimizer=None, provenance=None):
    # Only load trusted checkpoints produced by this project (optimizer/RNG objects).
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config_hash'] != config_hash:
        raise ValueError('Checkpoint configuration differs from this run')
    if provenance is not None and state['provenance'] != provenance:
        raise ValueError('Checkpoint data/autoencoder provenance differs')
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None and next(model.parameters()).device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def memory(device):
    if device.type != 'cuda':
        return {'device': str(device), 'peak_allocated_gib': None, 'peak_reserved_gib': None}
    torch.cuda.synchronize(device)
    return {'device': str(device), 'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30}


def append_metrics(path, values):
    with open(path, 'a') as handle:
        handle.write(json.dumps(values, allow_nan=False)+'\n')
    print(json.dumps(values, allow_nan=False), flush=True)


def ensure_finite_loss_and_step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)
