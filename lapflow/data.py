"""Read original NetCDF fields in bounded periodic crops; never coarsen DNS data."""
import json
from itertools import product
from pathlib import Path
import numpy as np
import torch
from netCDF4 import Dataset


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def discover(config):
    records, source_files, coordinates = {}, [], None
    all_times = {}
    for split, (lo, hi) in config['splits'].items():
        rows, seen = [], set()
        for file_id in range(lo, hi+1):
            path = Path(config['data_root'])/f'fluid_{file_id}.nc'
            stat = path.stat()
            with Dataset(path, 'r') as ds:
                for channel in config['channels']:
                    variable = ds[channel]
                    if variable.dimensions != ('time', 'x', 'y', 'z') or variable.shape[1:] != (config['grid'],)*3:
                        raise ValueError(f'Unexpected DNS shape/dimensions: {path}:{channel}')
                current = {axis: np.asarray(ds[axis][:], dtype=float).tolist() for axis in ('x', 'y', 'z')}
                if coordinates is None:
                    coordinates = current
                elif coordinates != current:
                    raise ValueError('Coordinate grids differ across source files')
                times = np.asarray(ds['time'][:], dtype=float)
                if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError('Invalid time coordinates: '+str(path))
                for index, time in enumerate(times):
                    if float(time) in seen:
                        continue  # Shared file-boundary snapshot, counted once per split.
                    if float(time) in all_times and all_times[float(time)] != split:
                        raise ValueError('Physical time appears in multiple data splits')
                    seen.add(float(time))
                    all_times[float(time)] = split
                    rows.append({'file': str(path), 'file_id': file_id, 'index': index, 'time': float(time)})
            source_files.append({'file': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        records[split] = rows
    return {'records': records, 'source_files': source_files, 'coordinates': coordinates}


def verify_sources(manifest):
    for source in manifest['source_files']:
        stat = Path(source['file']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source['size'], source['mtime_ns']):
            raise ValueError('Source changed since prepare: '+source['file'])


def periodic_blocks(origin, size, shape):
    """Contiguous source/destination slices, including windows larger than a grid."""
    if len(origin) != 3 or len(shape) != 3 or size < 1:
        raise ValueError('Expected three spatial axes and a positive window size')
    axes = []
    for start, extent in zip(origin, shape):
        segments, offset = [], 0
        while offset < size:
            source = (start+offset) % extent
            length = min(size-offset, extent-source)
            segments.append((slice(source, source+length), slice(offset, offset+length)))
            offset += length
        axes.append(segments)
    for block in product(*axes):
        yield tuple(pair[0] for pair in block), tuple(pair[1] for pair in block)


def read_window(record, origin, size, channels=('u', 'v', 'w')):
    # Never pass index arrays to NetCDF: periodic boundaries can cause millions of reads.
    values = np.empty((len(channels), size, size, size), dtype=np.float32)
    with Dataset(record['file'], 'r') as ds:
        ds.set_auto_mask(False)
        for channel_index, channel in enumerate(channels):
            variable = ds[channel]
            for source, target in periodic_blocks(origin, size, variable.shape[-3:]):
                values[(channel_index,)+target] = variable[(record['index'],)+source]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite DNS values: '+record['file'])
    return torch.from_numpy(values).unsqueeze(0)


def snapshot_reader(record, grid, stats):
    """Read one snapshot once on CPU; normalize bounded tiles, not the full volume."""
    volume = read_window(record, (0, 0, 0), grid)
    def reader(origin, size):
        window = torch.empty(1, volume.shape[1], size, size, size)
        for source, target in periodic_blocks(origin, size, volume.shape[-3:]):
            window[(...,)+target] = volume[(...,)+source]
        return normalize(window, stats)
    return reader


def normalize(x, stats):
    mean = torch.as_tensor(stats['mean'], dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1, 1)
    std = torch.as_tensor(stats['std'], dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1, 1)
    return (x-mean)/std


def restore(x, stats):
    mean = torch.as_tensor(stats['mean'], dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1, 1)
    std = torch.as_tensor(stats['std'], dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1, 1)
    return x*std+mean


class Moments:
    """Parallel Welford accumulation over channels; float64, bounded memory."""
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def add(self, x):
        values = x.detach().double().movedim(1, 0).reshape(x.shape[1], -1)
        count = values.shape[1]
        mean = values.mean(1)
        m2 = ((values-mean[:, None])**2).sum(1)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean-self.mean
        total = self.count+count
        self.m2 += m2+delta.square()*(self.count*count/total)
        self.mean += delta*(count/total)
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError('No samples for normalization')
        return {'mean': self.mean.tolist(), 'std': (self.m2/self.count).sqrt().clamp_min(1e-6).tolist(),
                'values_per_channel': self.count}


def random_crop(records, config, rng, stats=None, halo=True):
    record = records[int(rng.integers(len(records)))]
    a = config['autoencoder']
    factor = 2**len(a['widths'])
    origin = tuple(int(i)*factor for i in rng.integers(config['grid']//factor, size=3))
    padding = a['halo'] if halo else 0
    x = read_window(record, tuple(i-padding for i in origin), a['tile']+2*padding)
    return normalize(x, stats) if stats else x


def write_sample(path, volume, coordinates, metadata,
                 description='Unconditional generated velocity snapshot; no physical forecast time'):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.nc.tmp')
    if not torch.isfinite(volume).all():
        raise FloatingPointError('Nonfinite generated field')
    with Dataset(temporary, 'w') as ds:
        for axis in ('x', 'y', 'z'):
            ds.createDimension(axis, len(coordinates[axis]))
            ds.createVariable(axis, 'f4', (axis,))[:] = coordinates[axis]
        for i, channel in enumerate(('u', 'v', 'w')):
            ds.createVariable(channel, 'f4', ('x', 'y', 'z'))[:] = volume[0, i].numpy()
        ds.setncattr('description', description)
        ds.setncattr('provenance_json', json.dumps(metadata, sort_keys=True))
    temporary.replace(path)


def recover_volume(path, expected, coordinates):
    """Validate an atomic NetCDF output and recover a sidecar after interruption.

    Never overwrite an existing volume. Its embedded metadata is authoritative.
    """
    path = Path(path)
    with Dataset(path, 'r') as ds:
        metadata = json.loads(ds.provenance_json)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError('Existing volume has different provenance: '+str(path))
        shape = tuple(len(coordinates[axis]) for axis in ('x','y','z'))
        for channel in ('u','v','w'):
            if ds[channel].dimensions != ('x','y','z') or ds[channel].shape != shape:
                raise ValueError('Existing volume has incompatible dimensions')
        for axis in ('x','y','z'):
            if not np.array_equal(np.asarray(ds[axis][:]), np.asarray(coordinates[axis],dtype=np.float32)):
                raise ValueError('Existing volume has incompatible coordinates')
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != metadata:
            raise ValueError('Volume and sidecar disagree: '+str(path))
    else:
        atomic_json(sidecar, metadata)
    return metadata
