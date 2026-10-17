"""Validate configuration before allocating models or creating run artifacts."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate(config)
    return config


def validate(c):
    a, m, tr = c['autoencoder'], c['model'], c['training']
    factor = 2 ** len(a['widths'])
    if c['channels'] != ['u', 'v', 'w']:
        raise ValueError('This baseline generates u,v,w only')
    sizes = [c['grid'], a['tile'], a['halo'], a['latent_channels'], *a['widths'],
             m['width'], m['depth'], m['heads'], m['patch'], m['levels'], m['query_chunk']]
    if any(type(x) is not int or x <= 0 for x in sizes):
        raise ValueError('Sizes must be positive integers')
    if c['grid'] % a['tile'] or a['tile'] % factor or a['halo'] % factor or a['halo'] < 2*factor:
        raise ValueError('Grid must tile exactly; tile/halo align to compression; halo >= 2*compression')
    if m['width'] % m['heads'] or m['width'] % 6:
        raise ValueError('Width must be divisible by heads and six positional components')
    if m['levels'] not in (2, 3):
        raise ValueError('Only two or three levels supported')
    if c['grid'] % (factor*m['patch']*2**(m['levels']-1)):
        raise ValueError('Every latent pyramid level must be patch-divisible')
    starts = c['flow']['starts']
    if (len(starts) != m['levels'] or starts[0] != 0
            or not all(0 <= x < y <= 1 for x, y in zip(starts, starts[1:]+[1]))):
        raise ValueError('Stage starts must increase strictly from zero to below one')
    if set(c['splits']) != {'train', 'val', 'test'}:
        raise ValueError('Supply train, val, and test file ranges')
    ids = []
    for split, (lo, hi) in c['splits'].items():
        if type(lo) is not int or type(hi) is not int or lo > hi:
            raise ValueError('Invalid file range: '+split)
        ids.extend(range(lo, hi+1))
    if len(ids) != len(set(ids)):
        raise ValueError('File splits overlap')
    for key in ('ae_steps', 'flow_steps', 'save_every', 'stats_crops', 'batch_size', 'validation_samples'):
        if type(tr[key]) is not int or tr[key] <= 0:
            raise ValueError('Invalid training setting: '+key)
    if tr['lr'] <= 0 or c['flow']['steps_per_stage'] <= 0:
        raise ValueError('Learning rate and solver steps must be positive')
