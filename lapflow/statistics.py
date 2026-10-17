"""Held-out, unpaired volume statistics. All spatial FFT work runs on CPU.

Spectrum is shell-summed kinetic energy; sum over shells includes the mean mode
and equals 0.5*mean(u^2+v^2+w^2). Derivatives use periodic spectral differentiation
with the even-grid Nyquist derivative set to zero.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from netCDF4 import Dataset
from .config import fingerprint
from .data import atomic_json, verify_sources
from .runtime import sha256, autocast


METHOD_VERSION = 1
PDF_EDGES = np.linspace(-6., 6., 121)


def coordinate_spacing(coordinates):
    spacing = []
    for axis in ('x', 'y', 'z'):
        x = np.asarray(coordinates[axis], dtype=float)
        delta = np.diff(x)
        if len(delta) == 0 or not np.isfinite(x).all() or delta[0] == 0:
            raise ValueError('Invalid coordinate axis: '+axis)
        if not np.allclose(delta, delta.mean(), rtol=2e-4, atol=1e-8):
            raise ValueError('Spectral evaluation requires a uniform grid: '+axis)
        spacing.append(float(delta.mean()))
    return spacing


def read_volume(path, index=None):
    with Dataset(path, 'r') as ds:
        ds.set_auto_mask(False)
        arrays = []
        for name in ('u', 'v', 'w'):
            variable = ds[name]
            expected = ('x', 'y', 'z') if index is None else ('time', 'x', 'y', 'z')
            if variable.dimensions != expected:
                raise ValueError('Unexpected volume dimensions: '+str(path))
            arrays.append(np.asarray(variable[:] if index is None else variable[index], dtype=np.float32))
    x = torch.from_numpy(np.stack(arrays))
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite volume: '+str(path))
    return x


@torch.no_grad()
def volume_statistics(velocity, spacing, normalization):
    """velocity: CPU [3,X,Y,Z], coordinate spacings may have either orientation."""
    if velocity.device.type != 'cpu' or velocity.ndim != 4 or velocity.shape[0] != 3:
        raise ValueError('Statistics require one CPU three-component volume')
    if not torch.isfinite(velocity).all():
        raise FloatingPointError('Nonfinite velocity')
    sizes = velocity.shape[1:]
    dtype = velocity.dtype
    k = [2*math.pi*torch.fft.fftfreq(sizes[i], d=abs(spacing[i]), dtype=dtype) for i in (0, 1)]
    k.append(2*math.pi*torch.fft.rfftfreq(sizes[2], d=abs(spacing[2]), dtype=dtype))
    dk = min(2*math.pi/(n*abs(dx)) for n, dx in zip(sizes, spacing))
    radii = torch.sqrt(k[0][:, None, None].square()+k[1][None, :, None].square()+k[2][None, None, :].square())
    shells = torch.floor(radii/dk+.5).long()
    shell_count = int(shells.max())+1
    del radii
    # Restore negative-z conjugate energy omitted by rfftn.
    weights = torch.full((sizes[2]//2+1,), 2., dtype=dtype)
    weights[0] = 1
    if sizes[2] % 2 == 0:
        weights[-1] = 1
    derivative_k = []
    for i in range(3):
        axis = k[i].clone()*np.sign(spacing[i])
        if sizes[i] % 2 == 0:
            axis[sizes[i]//2] = 0
        derivative_k.append(axis)
    gradient_k2 = (derivative_k[0][:, None, None].square()
                   + derivative_k[1][None, :, None].square()
                   + derivative_k[2][None, None, :].square())
    spectrum = torch.zeros(shell_count, dtype=torch.float64)
    divergence_hat = None
    gradient_energy = energy = 0.
    means, rms, pdf, tails = [], [], [], []
    for component in range(3):
        field = velocity[component]
        means.append(float(field.double().mean()))
        mean_square = float(field.double().square().mean())
        rms.append(math.sqrt(mean_square))
        energy += .5*mean_square
        scaled = (field-normalization['mean'][component])/normalization['std'][component]
        counts = torch.histc(scaled.float(), bins=len(PDF_EDGES)-1,
                             min=float(PDF_EDGES[0]), max=float(PDF_EDGES[-1])).double()
        pdf.append((counts/(field.numel()*np.diff(PDF_EDGES)[0])).tolist())
        tails.append(max(0., 1.-float(counts.sum())/field.numel()))
        del scaled, counts
        transform = torch.fft.rfftn(field, norm='forward')
        power = transform.abs().square()*weights
        spectrum += .5*torch.bincount(shells.flatten(), weights=power.flatten().double(), minlength=shell_count)
        gradient_energy += float((gradient_k2*power).double().sum())
        shape = [1, 1, 1]
        shape[component] = -1
        contribution = 1j*derivative_k[component].reshape(shape)*transform
        if divergence_hat is None:
            divergence_hat = contribution
        else:
            divergence_hat += contribution
        del transform, contribution, power
    divergence_energy = float((divergence_hat.abs().square()*weights).double().sum())
    divergence_rms = math.sqrt(max(divergence_energy, 0.))
    return {'kinetic_energy': energy, 'mean_velocity': means, 'rms_velocity': rms,
            'spectrum_k': (np.arange(shell_count)*dk).tolist(), 'spectrum_shell_energy': spectrum.tolist(),
            'spectrum_energy_sum': float(spectrum.sum()),
            'parseval_relative_error': abs(float(spectrum.sum())-energy)/max(energy, 1e-30),
            'divergence_rms': divergence_rms, 'velocity_gradient_rms': math.sqrt(max(gradient_energy, 0.)),
            'relative_divergence': math.sqrt(divergence_energy/gradient_energy) if gradient_energy > 0 else 0.,
            'pdf_edges': PDF_EDGES.tolist(), 'normalized_velocity_pdf': pdf, 'pdf_outside_fraction': tails}


def cached_statistics(folder, key, path, index, coordinates, stats):
    stat = Path(path).stat()
    provenance = {'method': METHOD_VERSION, 'path': str(path), 'index': index, 'size': stat.st_size,
                  'mtime_ns': stat.st_mtime_ns, 'coordinates': fingerprint(coordinates), 'normalization': stats}
    destination = folder/(key+'.json')
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved['provenance'] != provenance:
            raise ValueError('Evaluation cache provenance changed: '+str(destination))
        return saved['statistics']
    value = volume_statistics(read_volume(path, index), coordinate_spacing(coordinates), stats)
    atomic_json(destination, {'provenance': provenance, 'statistics': value})
    return value


def aggregate(rows):
    keys = ('spectrum_shell_energy', 'normalized_velocity_pdf', 'pdf_outside_fraction')
    result = {key: np.mean([row[key] for row in rows], axis=0).tolist() for key in keys}
    result.update(count=len(rows), spectrum_k=rows[0]['spectrum_k'], pdf_edges=rows[0]['pdf_edges'])
    for key in ('kinetic_energy', 'divergence_rms', 'relative_divergence'):
        values = np.asarray([row[key] for row in rows])
        result[key] = {'mean': float(values.mean()), 'std': float(values.std()), 'values': values.tolist()}
    result['max_parseval_relative_error'] = max(row['parseval_relative_error'] for row in rows)
    return result


@torch.no_grad()
def evaluate_test_flow(config, run, device, cache, flow_hash):
    from .train import latent_manifest, make_flow, read_latent
    from .runtime import load_checkpoint
    from .flow import loss
    _, manifest = latent_manifest(config, run)
    model = make_flow(config).to(device).eval()
    load_checkpoint(run/'flow'/'best.pt', model, fingerprint(config), provenance=fingerprint(manifest))
    rows = []
    for i, record in enumerate(manifest['records']['test']):
        path = cache/f'test-flow-{i}.json'
        provenance = {'latent_manifest': fingerprint(manifest), 'flow_sha256': flow_hash,
                      'method': METHOD_VERSION, 'record': record, 'rng_seed': config['seed']+500000+i}
        if path.exists():
            row = json.loads(path.read_text())
            if row['provenance'] != provenance:
                raise ValueError('Test flow evaluation cache changed')
        else:
            generator = torch.Generator(device=device).manual_seed(provenance['rng_seed'])
            z = read_latent(run, record, manifest['stats']).to(device)
            with autocast(device, config['training']['amp']):
                values = [float(loss(model, z, config['flow']['starts'], active, generator))
                          for active in range(1, config['model']['levels']+1)]
            row = {'provenance': provenance, 'loss_by_active_scale_count': values}
            atomic_json(path, row)
        rows.append(row['loss_by_active_scale_count'])
    values = np.asarray(rows)
    return {'snapshots': len(rows), 'mean_loss': float(values.mean()),
            'mean_by_active_scale_count': values.mean(axis=0).tolist(),
            'description': 'Fixed-noise Monte Carlo transport MSE, all test latents, each active scale count'}


def evaluate_experiment(config, run, device, seeds):
    from .train import run_manifest, latent_manifest
    from .report import write_report
    manifest = run_manifest(config, run)
    verify_sources(manifest)
    _, latent = latent_manifest(config, run)
    folder = run/'evaluation'
    cache = folder/'per-volume'
    cache.mkdir(parents=True, exist_ok=True)
    flow_hash = sha256(run/'flow'/'best.pt')
    test_loss = evaluate_test_flow(config, run, device, cache, flow_hash)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    dns_rows, generated_rows = [], []
    records = manifest['records']['test']
    for i, record in enumerate(records):
        dns_rows.append(cached_statistics(cache, f'dns-{i}', record['file'], record['index'],
                                         manifest['coordinates'], manifest['velocity_stats']))
        print(f'Test statistics: {i+1}/{len(records)}', flush=True)
    for seed in seeds:
        path = run/f'sample-{seed}.nc'
        with Dataset(path, 'r') as ds:
            provenance = json.loads(ds.provenance_json)
            if (provenance['seed'] != seed or provenance['config'] != config
                    or provenance['flow_sha256'] != flow_hash or provenance['latent_provenance'] != latent['provenance']):
                raise ValueError('Generated sample does not match evaluated model: '+str(path))
        generated_rows.append(cached_statistics(cache, f'generated-{seed}', path, None,
                                               manifest['coordinates'], manifest['velocity_stats']))
        print(f'Generated statistics: seed {seed}', flush=True)
    dns, generated = aggregate(dns_rows), aggregate(generated_rows)
    target_spectrum = np.asarray(dns['spectrum_shell_energy'])
    spectrum = np.asarray(generated['spectrum_shell_energy'])
    energy = dns['kinetic_energy']['mean']
    pdf_width = np.diff(PDF_EDGES)[0]
    result = {'method_version': METHOD_VERSION, 'config_hash': fingerprint(config), 'flow_sha256': flow_hash,
              'autoencoder_sha256': latent['provenance']['autoencoder_sha256'], 'sample_seeds': seeds,
              'test_source_records': records, 'test_flow_loss': test_loss, 'dns': dns, 'generated': generated,
              'comparison': {'mean_kinetic_energy_relative_bias': (generated['kinetic_energy']['mean']-energy)/max(energy, 1e-30),
                             'spectrum_relative_l1_excluding_mean': float(np.abs(spectrum[1:]-target_spectrum[1:]).sum()/max(target_spectrum[1:].sum(), 1e-30)),
                             'velocity_pdf_l1_by_component': (np.abs(np.asarray(generated['normalized_velocity_pdf'])-np.asarray(dns['normalized_velocity_pdf'])).sum(axis=1)*pdf_width).tolist()},
              'limitations': ['Unconditional samples are not paired forecasts; no generated-vs-DNS pointwise MSE.',
                              'Generated sample count is small and DNS snapshots may be temporally correlated.',
                              'No confidence intervals or claim of physical validity.',
                              'Periodic spectral derivatives set even-grid Nyquist derivatives to zero.',
                              'Velocity PDFs use training normalization, range [-6,6]; excluded tail fractions reported.']}
    reconstruction = run/'reconstruction-test-0.json'
    if reconstruction.exists():
        result['test_autoencoder_reconstruction'] = json.loads(reconstruction.read_text())
    atomic_json(folder/'metrics.json', result)
    write_report(folder, result)
    return result
