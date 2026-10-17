"""Full-volume reconstruction assessment before trusting the learned compression."""
import json
from itertools import product
import torch
from .autoencoder import encode_tiled, decode_tiled
from .config import fingerprint
from .data import read_window, normalize, restore, write_sample, atomic_json, recover_volume, snapshot_reader
from .runtime import load_checkpoint, autocast, memory, sha256


def reconstruct_volume(config, run, device, split='val', index=0, resume=False):
    from .train import run_manifest, make_autoencoder
    manifest = run_manifest(config, run)
    record = manifest['records'][split][index]
    checkpoint_path = run/'autoencoder'/'best.pt'
    output = run/f'reconstruction-{split}-{index}.nc'
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        recover_volume(output, {'source': record, 'autoencoder_sha256': sha256(checkpoint_path)}, manifest['coordinates'])
        print('Verified existing reconstruction: '+str(output), flush=True)
        return
    model = make_autoencoder(config).to(device).eval()
    load_checkpoint(checkpoint_path, model, fingerprint(config), provenance=fingerprint(manifest))
    stats = manifest['velocity_stats']
    a = config['autoencoder']
    reader = snapshot_reader(record, config['grid'], stats)
    with autocast(device, config['training']['amp']):
        latent = encode_tiled(model, reader, config['grid'], a['tile'], a['halo'], device)
        del reader
        prediction = restore(decode_tiled(model, latent, a['tile'], a['halo'], device), stats)
    squared_error = normalized_error = true_energy = predicted_energy = count = 0
    for origin in product(range(0, config['grid'], a['tile']), repeat=3):
        actual = read_window(record, origin, a['tile'])
        slices = tuple(slice(i,i+a['tile']) for i in origin)
        predicted = prediction[(...,)+slices]
        squared_error += float((actual.double()-predicted.double()).square().sum())
        normalized_error += float((normalize(actual,stats).double()-normalize(predicted,stats).double()).square().sum())
        true_energy += float(actual.double().square().sum())
        predicted_energy += float(predicted.double().square().sum())
        count += actual.numel()
    metrics = {'source': record, 'physical_mse': squared_error/count,
               'normalized_mse': normalized_error/count,
               'relative_squared_l2_error': squared_error/max(true_energy,1e-30),
               'actual_mean_kinetic_energy': 1.5*true_energy/count,
               'reconstructed_mean_kinetic_energy': 1.5*predicted_energy/count,
               'autoencoder_sha256': sha256(checkpoint_path), **memory(device)}
    output = run/f'reconstruction-{split}-{index}.nc'
    write_sample(output,prediction,manifest['coordinates'],metrics,
                 description='Autoencoder reconstruction of a recorded DNS snapshot; not a generated future')
    atomic_json(output.with_suffix('.json'),metrics)
    print(json.dumps(metrics,indent=2),flush=True)
