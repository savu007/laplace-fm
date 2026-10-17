"""Explicit stages: metadata/stats, local autoencoder, frozen cache, global flow.

All artifacts live in the requested run directory; DNS inputs are read-only.
No job submission, GPU selection by fallback, or automatic resolution changes.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .autoencoder import Autoencoder3D, core, encode_tiled, decode_tiled
from .config import load_config, fingerprint
from .data import (atomic_json, discover, verify_sources, read_window, normalize, restore,
                   Moments, random_crop, write_sample, recover_volume, snapshot_reader)
from .model import LapFlow3D
from .flow import loss as flow_loss, sample as flow_sample
from .runtime import (seed_all, device_for, autocast, sha256, save_checkpoint,
                      load_checkpoint, memory, append_metrics, ensure_finite_loss_and_step)


def make_autoencoder(c):
    a = c['autoencoder']
    return Autoencoder3D(a['widths'], a['latent_channels'])


def make_flow(c):
    return LapFlow3D(latent_channels=c['autoencoder']['latent_channels'], **c['model'])


def run_manifest(c, run):
    manifest = json.loads((run/'dataset.json').read_text())
    if manifest['config_hash'] != fingerprint(c):
        raise ValueError('Run configuration differs; use the original config or a new run directory')
    return manifest


def prepare(c, run):
    if (run/'dataset.json').exists():
        manifest = run_manifest(c, run)
        verify_sources(manifest)
        print('Existing prepared dataset verified', flush=True)
        return
    manifest = discover(c)
    moments = Moments()
    rng = np.random.default_rng(c['seed'])
    for _ in range(c['training']['stats_crops']):
        moments.add(random_crop(manifest['records']['train'], c, rng, halo=False))
    manifest.update(config_hash=fingerprint(c), config=c, velocity_stats=moments.result(),
                    normalization_method='training-only random aligned periodic crops')
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run/'config.json', c)
    atomic_json(run/'dataset.json', manifest)
    print({s: len(r) for s, r in manifest['records'].items()}, flush=True)


def validation_ae(model, records, c, stats, device):
    model.eval()
    rng = np.random.default_rng(c['seed']+99)
    values = []
    with torch.no_grad(), autocast(device, c['training']['amp']):
        for _ in range(c['training']['validation_samples']):
            x = random_crop(records, c, rng, stats).to(device)
            y = model(x)
            a = c['autoencoder']
            values.append(float((core(y, a['halo'], a['tile']).float()-core(x, a['halo'], a['tile'])).square().mean()))
    model.train()
    return float(np.mean(values))


def train_ae(c, run, device, resume=False):
    manifest = run_manifest(c, run)
    verify_sources(manifest)
    model = make_autoencoder(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=0.)
    folder = run/'autoencoder'
    folder.mkdir(exist_ok=True)
    path = folder/'last.pt'
    provenance = fingerprint(manifest)
    start, best = 0, float('inf')
    if path.exists():
        if not resume:
            raise FileExistsError('Use --resume for existing autoencoder training')
        state = load_checkpoint(path, model, fingerprint(c), optimizer, provenance)
        start, best = state['step'], state['best']
    a, tr = c['autoencoder'], c['training']
    # Recovery checkpoints precede validation; finish selection on resume too.
    if start:
        val = validation_ae(model, manifest['records']['val'], c, manifest['velocity_stats'], device)
        if val < best or not (folder/'best.pt').exists():
            best = min(best, val)
            save_checkpoint(folder/'best.pt', model, optimizer, start, best, fingerprint(c), provenance)
        save_checkpoint(path, model, optimizer, start, best, fingerprint(c), provenance)
    last_saved = time.monotonic()
    print(f'Training ae: starting at step {start}; target {tr["ae_steps"]}', flush=True)
    for step in range(start+1, tr['ae_steps']+1):
        step_started = time.monotonic()
        # Step-indexed crop RNG permits exact data selection after resume.
        rng = np.random.default_rng(c['seed']+step)
        x = torch.cat([random_crop(manifest['records']['train'], c, rng, manifest['velocity_stats'])
                       for _ in range(tr['batch_size'])]).to(device)
        data_seconds = time.monotonic()-step_started
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, tr['amp']):
            y = model(x)
            loss = (core(y, a['halo'], a['tile']).float()-core(x, a['halo'], a['tile'])).square().mean()
        ensure_finite_loss_and_step(loss, model, optimizer)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        step_seconds = time.monotonic()-step_started
        validate_now = step % tr['save_every'] == 0 or step == tr['ae_steps']
        checkpoint_now = step == start+1 or validate_now or time.monotonic()-last_saved >= 60
        if checkpoint_now:
            save_checkpoint(path, model, optimizer, step, best, fingerprint(c), provenance)
            last_saved = time.monotonic()
            print(f'[checkpoint] ae step {step}: {path}', flush=True)
        if step <= start+3 or step % 10 == 0 or checkpoint_now:
            append_metrics(folder/'progress.jsonl', {'step': step, 'train_normalized_mse': float(loss.detach()),
                           'data_seconds': data_seconds, 'step_seconds': step_seconds,
                           'checkpoint_saved': checkpoint_now, **memory(device)})
        if validate_now:
            print(f'[validation] ae step {step}', flush=True)
            val = validation_ae(model, manifest['records']['val'], c, manifest['velocity_stats'], device)
            improved = val < best
            best = min(best, val)
            # The recovery checkpoint already contains this completed update.
            # Resume repeats validation if selection was interrupted.
            if improved:
                save_checkpoint(folder/'best.pt', model, optimizer, step, best, fingerprint(c), provenance)
            save_checkpoint(path, model, optimizer, step, best, fingerprint(c), provenance)
            append_metrics(folder/'metrics.jsonl', {'step': step, 'train_normalized_mse': float(loss.detach()),
                                                   'val_normalized_mse': val, **memory(device)})


def cache_latents(c, run, device):
    manifest = run_manifest(c, run)
    verify_sources(manifest)
    checkpoint_path = run/'autoencoder'/'best.pt'
    provenance = {'dataset': fingerprint(manifest), 'autoencoder_sha256': sha256(checkpoint_path)}
    folder = run/'latents'
    folder.mkdir(exist_ok=True)
    if (folder/'provenance.json').exists():
        if json.loads((folder/'provenance.json').read_text()) != provenance:
            raise ValueError('Latent cache belongs to different data/weights; use a new run')
    else:
        atomic_json(folder/'provenance.json', provenance)
    model = make_autoencoder(c).to(device).eval()
    load_checkpoint(checkpoint_path, model, fingerprint(c), provenance=fingerprint(manifest))
    a = c['autoencoder']
    moments, rows = Moments(), {}
    for split, records in manifest['records'].items():
        rows[split] = []
        for number, record in enumerate(records):
            name = f"{split}-{record['file_id']}-{record['index']}.npy"
            path = folder/name
            if path.exists():
                latent = torch.from_numpy(np.load(path, allow_pickle=False).copy())
            else:
                reader = snapshot_reader(record, c['grid'], manifest['velocity_stats'])
                with autocast(device, c['training']['amp']):
                    latent = encode_tiled(model, reader, c['grid'], a['tile'], a['halo'], device)
                del reader
                expected = (1, a['latent_channels'], *([c['grid']//model.factor]*3))
                if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
                    raise ValueError('Invalid encoded latent: '+str(path))
                temporary = path.with_suffix('.npy.tmp')
                with open(temporary, 'wb') as handle:
                    np.save(handle, latent.numpy())
                temporary.replace(path)
            expected = (1, a['latent_channels'], *([c['grid']//model.factor]*3))
            if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
                raise ValueError('Invalid cached latent: '+str(path))
            if split == 'train':
                moments.add(latent)
            rows[split].append({'path': name, 'source': record})
            if number % 10 == 0:
                print(f'Encoded {split}: {number+1}/{len(records)}', flush=True)
    atomic_json(folder/'manifest.json', {'provenance': provenance, 'records': rows, 'stats': moments.result()})


def latent_manifest(c, run):
    data = run_manifest(c, run)
    latent = json.loads((run/'latents'/'manifest.json').read_text())
    expected = {'dataset': fingerprint(data), 'autoencoder_sha256': sha256(run/'autoencoder'/'best.pt')}
    if latent['provenance'] != expected:
        raise ValueError('Cache is stale relative to dataset/autoencoder')
    return data, latent


def read_latent(run, record, stats):
    return normalize(torch.from_numpy(np.load(run/'latents'/record['path'], allow_pickle=False).copy()), stats)


def validation_flow(model, c, run, manifest, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(c['seed']+100)
    values = []
    with torch.no_grad(), autocast(device, c['training']['amp']):
        for record in manifest['records']['val'][:c['training']['validation_samples']]:
            z = read_latent(run, record, manifest['stats']).to(device)
            for active in range(1, c['model']['levels']+1):
                values.append(float(flow_loss(model, z, c['flow']['starts'], active, generator)))
    model.train()
    return float(np.mean(values))


def train_flow(c, run, device, resume=False):
    _, manifest = latent_manifest(c, run)
    model = make_flow(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=0.)
    folder = run/'flow'
    folder.mkdir(exist_ok=True)
    path = folder/'last.pt'
    provenance = fingerprint(manifest)
    start, best = 0, float('inf')
    if path.exists():
        if not resume:
            raise FileExistsError('Use --resume for existing flow training')
        state = load_checkpoint(path, model, fingerprint(c), optimizer, provenance)
        start, best = state['step'], state['best']
    tr = c['training']
    # Recovery checkpoints precede validation; finish selection on resume too.
    if start:
        val = validation_flow(model, c, run, manifest, device)
        if val < best or not (folder/'best.pt').exists():
            best = min(best, val)
            save_checkpoint(folder/'best.pt', model, optimizer, start, best, fingerprint(c), provenance)
        save_checkpoint(path, model, optimizer, start, best, fingerprint(c), provenance)
    last_saved = time.monotonic()
    print(f'Training flow: starting at step {start}; target {tr["flow_steps"]}', flush=True)
    for step in range(start+1, tr['flow_steps']+1):
        step_started = time.monotonic()
        rng = np.random.default_rng(c['seed']+step)
        records = manifest['records']['train']
        z = torch.cat([read_latent(run, records[int(rng.integers(len(records)))], manifest['stats'])
                       for _ in range(tr['batch_size'])]).to(device)
        data_seconds = time.monotonic()-step_started
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, tr['amp']):
            loss = flow_loss(model, z, c['flow']['starts'])
        ensure_finite_loss_and_step(loss, model, optimizer)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        step_seconds = time.monotonic()-step_started
        validate_now = step % tr['save_every'] == 0 or step == tr['flow_steps']
        checkpoint_now = step == start+1 or validate_now or time.monotonic()-last_saved >= 60
        if checkpoint_now:
            save_checkpoint(path, model, optimizer, step, best, fingerprint(c), provenance)
            last_saved = time.monotonic()
            print(f'[checkpoint] flow step {step}: {path}', flush=True)
        if step <= start+3 or step % 10 == 0 or checkpoint_now:
            append_metrics(folder/'progress.jsonl', {'step': step, 'train_flow_loss': float(loss.detach()),
                           'data_seconds': data_seconds, 'step_seconds': step_seconds,
                           'checkpoint_saved': checkpoint_now, **memory(device)})
        if validate_now:
            print(f'[validation] flow step {step}', flush=True)
            val = validation_flow(model, c, run, manifest, device)
            improved = val < best
            best = min(best, val)
            # The recovery checkpoint already contains this completed update.
            # Resume repeats validation if selection was interrupted.
            if improved:
                save_checkpoint(folder/'best.pt', model, optimizer, step, best, fingerprint(c), provenance)
            save_checkpoint(path, model, optimizer, step, best, fingerprint(c), provenance)
            append_metrics(folder/'metrics.jsonl', {'step': step, 'train_flow_loss': float(loss.detach()),
                                                   'val_flow_loss': val, **memory(device)})


def generate(c, run, device, seed, resume=False):
    data, latent = latent_manifest(c, run)
    output = run/f'sample-{seed}.nc'
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        recover_volume(output, {'seed': seed, 'config': c,
                                'flow_sha256': sha256(run/'flow'/'best.pt'),
                                'latent_provenance': latent['provenance']}, data['coordinates'])
        print('Verified existing sample: '+str(output), flush=True)
        return
    model = make_flow(c).to(device).eval()
    checkpoint_path = run/'flow'/'best.pt'
    load_checkpoint(checkpoint_path, model, fingerprint(c), provenance=fingerprint(latent))
    factor = 2**len(c['autoencoder']['widths'])
    shape = (1, c['autoencoder']['latent_channels'], *([c['grid']//factor]*3))
    generator = torch.Generator(device=device).manual_seed(seed)
    with autocast(device, c['training']['amp']):
        z = flow_sample(model, shape, c['flow']['starts'], c['flow']['steps_per_stage'], device, generator)
    z = restore(z.float().cpu(), latent['stats'])
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    autoencoder = make_autoencoder(c).to(device).eval()
    load_checkpoint(run/'autoencoder'/'best.pt', autoencoder, fingerprint(c), provenance=fingerprint(data))
    a = c['autoencoder']
    with autocast(device, c['training']['amp']):
        volume = decode_tiled(autoencoder, z, a['tile'], a['halo'], device)
    volume = restore(volume, data['velocity_stats'])
    metadata = {'seed': seed, 'config': c, 'flow_sha256': sha256(checkpoint_path),
                'latent_provenance': latent['provenance'], **memory(device)}
    write_sample(output, volume, data['coordinates'], metadata)
    atomic_json(output.with_suffix('.json'), metadata)
    print(str(output), flush=True)


def smoke(c, device):
    """Full configured tile and latent geometry, synthetic data, no DNS reads."""
    tr, a = c['training'], c['autoencoder']
    reports = {}
    for stage in ('autoencoder', 'flow'):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model = (make_autoencoder(c) if stage == 'autoencoder' else make_flow(c)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'])
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        with autocast(device, tr['amp']):
            if stage == 'autoencoder':
                side = a['tile']+2*a['halo']
                x = torch.randn(tr['batch_size'], 3, side, side, side, device=device)
                y = model(x)
                loss = (core(y, a['halo'], a['tile']).float()-core(x, a['halo'], a['tile'])).square().mean()
            else:
                side = c['grid']//2**len(a['widths'])
                x = torch.randn(tr['batch_size'], a['latent_channels'], side, side, side, device=device)
                loss = flow_loss(model, x, c['flow']['starts'], active=c['model']['levels'])
        ensure_finite_loss_and_step(loss, model, optimizer)
        reports[stage] = {'parameters': sum(p.numel() for p in model.parameters()), 'loss': float(loss.detach()),
                          'seconds': time.monotonic()-start, **memory(device)}
        del model, optimizer, loss, x
        if stage == 'autoencoder':
            del y
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(reports, indent=2), flush=True)
    return reports


def real_data_preflight(c, run, device):
    """Exercise DNS reads, optimizer steps, validation and one full-volume encoding.

    Disposable weights only; this does not count toward training or select a model.
    Pipeline reseeds before the actual training stage.
    """
    manifest = run_manifest(c, run)
    model = make_autoencoder(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=0.)
    a = c['autoencoder']
    side = a['tile']+2*a['halo']
    record = manifest['records']['train'][0]
    reports = []
    for origin in ((0, 0, 0), (-a['halo'], 0, 0), (-a['halo'],)*3):
        started = time.monotonic()
        x = normalize(read_window(record, origin, side), manifest['velocity_stats'])
        x = x.repeat(c['training']['batch_size'], 1, 1, 1, 1).to(device)
        data_seconds = time.monotonic()-started
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, c['training']['amp']):
            loss = (core(model(x), a['halo'], a['tile']).float()-core(x, a['halo'], a['tile'])).square().mean()
        ensure_finite_loss_and_step(loss, model, optimizer)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        report = {'origin': origin, 'data_seconds': data_seconds,
                  'step_seconds': time.monotonic()-started, 'loss': float(loss.detach()), **memory(device)}
        print('[real-data preflight] '+json.dumps(report), flush=True)
        reports.append(report)
    started = time.monotonic()
    val = validation_ae(model, manifest['records']['val'], c, manifest['velocity_stats'], device)
    validation_seconds = time.monotonic()-started
    started = time.monotonic()
    reader = snapshot_reader(record, c['grid'], manifest['velocity_stats'])
    with autocast(device, c['training']['amp']):
        latent = encode_tiled(model.eval(), reader, c['grid'], a['tile'], a['halo'], device)
    if not torch.isfinite(latent).all():
        raise FloatingPointError('Nonfinite preflight latent')
    encode_seconds = time.monotonic()-started
    result = {'crops': reports, 'validation_mse': val, 'validation_seconds': validation_seconds,
              'full_snapshot_encode_seconds': encode_seconds,
              'rough_ae_training_hours': np.mean([r['step_seconds'] for r in reports])*c['training']['ae_steps']/3600,
              'rough_cache_hours': encode_seconds*sum(len(r) for r in manifest['records'].values())/3600,
              'caution': 'Tiny timing sample; excludes flow training, sampling, evaluation and filesystem variation',
              **memory(device)}
    atomic_json(run/'real-data-preflight.json', result)
    print('[real-data preflight] '+json.dumps(result), flush=True)
    return result


def estimate(c):
    factor = 2**len(c['autoencoder']['widths'])
    sides = [c['grid']//factor//2**i for i in reversed(range(c['model']['levels']))]
    tokens = [(s//c['model']['patch'])**3 for s in sides]
    print(json.dumps({'dns_grid': c['grid'], 'latent_grids_coarse_first': sides, 'tokens': tokens,
                      'total_tokens': sum(tokens), 'fp32_velocity_volume_gib': 3*c['grid']**3*4/2**30,
                      'dense_masks_allocated': False,
                      'largest_math_attention_chunk_fp32_mib': c['training']['batch_size']*c['model']['heads']*
                      min(c['model']['query_chunk'], max(tokens))*sum(tokens)*4/2**20,
                      'warning': 'Tensor arithmetic, not measured peak GPU memory'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'autoencoder', 'cache', 'flow', 'sample', 'reconstruct', 'smoke', 'estimate'])
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent.parent/'configs/dns384.json'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    c = load_config(args.config)
    torch.set_num_threads(args.threads)
    if args.stage == 'estimate':
        estimate(c)
        return
    seed_all(c['seed'])
    device = device_for(args.device)
    if args.stage == 'smoke':
        smoke(c, device)
        return
    if args.run is None:
        parser.error('--run is required for data/training/sampling stages')
    from .evaluate import reconstruct_volume
    actions = {'reconstruct': lambda: reconstruct_volume(c, args.run, device, args.split, args.index, args.resume),
               'prepare': lambda: prepare(c, args.run),
               'autoencoder': lambda: train_ae(c, args.run, device, args.resume),
               'cache': lambda: cache_latents(c, args.run, device),
               'flow': lambda: train_flow(c, args.run, device, args.resume),
               'sample': lambda: generate(c, args.run, device, args.seed, args.resume)}
    actions[args.stage]()


if __name__ == '__main__':
    main()
