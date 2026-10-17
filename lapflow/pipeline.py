"""Sequential full experiment; explicit stage status and conservative resumption."""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import platform
import time
import torch
from .config import load_config, fingerprint
from .data import atomic_json, verify_sources
from .runtime import seed_all, device_for, sha256
from .train import prepare, train_ae, cache_latents, train_flow, generate, smoke, real_data_preflight
from .evaluate import reconstruct_volume
from .statistics import evaluate_experiment


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hash():
    return fingerprint({path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))})


def artifact_stamp(path):
    stat = path.stat()
    # Strong hashes for metadata; large fields are identified by size/mtime and
    # embedded provenance. Avoid rereading multi-GB artifacts solely for restart.
    value = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if stat.st_size < 1024*1024:
        value['sha256'] = sha256(path)
    return value


def import_prepared(config, run, source):
    """Explicitly reuse only verified metadata/statistics, never old training state."""
    source, run = Path(source).resolve(), Path(run).resolve()
    if source == run:
        raise ValueError('Prepared source and new run must differ')
    manifest_path = source/'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config_hash'] != fingerprint(config) or manifest['config'] != config:
        raise ValueError('Prepared source configuration differs')
    status = json.loads((source/'pipeline-status.json').read_text())
    prepared = status['stages']['prepare']
    if prepared['state'] != 'complete':
        raise ValueError('Source preparation did not finish')
    for name in ('dataset.json', 'config.json'):
        if artifact_stamp(source/name) != prepared['artifacts'][name]:
            raise ValueError('Prepared source artifact changed: '+name)
    verify_sources(manifest)
    receipt = {'source_run': str(source), 'dataset_sha256': sha256(manifest_path),
               'source_plan': status['plan'], 'scope': 'dataset metadata and normalization only'}
    if (run/'prepared-import.json').exists():
        if json.loads((run/'prepared-import.json').read_text()) != receipt:
            raise ValueError('Prepared import source changed')
    for name, value in (('dataset.json', manifest), ('config.json', config)):
        path = run/name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('Destination prepared metadata differs')
        else:
            atomic_json(path, value)
    atomic_json(run/'prepared-import.json', receipt)
    print('Reused verified prepared dataset from '+str(source), flush=True)


def experiment(config, run, device, samples=8, seed_base=50000, prepared_from=None):
    if samples < 1 or seed_base < 0:
        raise ValueError('Positive sample count and nonnegative seed required')
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed_base, seed_base+samples))
    plan = {'config_hash': fingerprint(config), 'source_hash': source_hash(), 'sample_seeds': seeds,
            'test_selection': 'all recorded test snapshots', 'evaluation_version': 1}
    status_path = run/'pipeline-status.json'
    # Prevent two submitted jobs from writing the same checkpoints/cache.
    with open(run/'.pipeline.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another full-stack job owns this run directory') from error
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['plan'] != plan:
                raise ValueError('Configuration, source, or evaluation plan changed; use a fresh RUN_DIR')
        else:
            status = {'plan': plan, 'created_utc': now(), 'stages': {}}
        status['last_started_utc'] = now()
        status['environment'] = {'host': platform.node(), 'torch': torch.__version__,
                                 'device': str(device), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
        status['state'] = 'running'
        atomic_json(status_path, status)
        if (run/'dataset.json').exists():
            try:
                verify_sources(json.loads((run/'dataset.json').read_text()))
            except Exception as error:
                status.update(state='failed', active_stage='source-verification', error=str(error))
                atomic_json(status_path, status)
                raise

        def stage(name, action, outputs, always=False):
            previous = status['stages'].get(name)
            if previous and previous.get('state') == 'complete' and not always:
                for relative, stamp in previous['artifacts'].items():
                    path = run/relative
                    if not path.exists() or artifact_stamp(path) != stamp:
                        error = 'Completed stage artifact is missing or changed: '+str(path)
                        status.update(state='failed', active_stage=name)
                        status['stages'][name].update(state='failed', error=error, finished_utc=now())
                        atomic_json(status_path, status)
                        raise ValueError(error)
                print(f'[skip] {name}: verified complete', flush=True)
                return
            print(f'[start] {name}', flush=True)
            status['active_stage'] = name
            status['stages'][name] = {'state': 'running', 'started_utc': now()}
            atomic_json(status_path, status)
            start = time.monotonic()
            try:
                seed_all(config['seed'])
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                action()
                artifacts = {relative: artifact_stamp(run/relative) for relative in outputs}
                status['stages'][name].update(state='complete', finished_utc=now(),
                                              seconds=time.monotonic()-start, artifacts=artifacts)
                atomic_json(status_path, status)
                print(f'[done] {name}', flush=True)
            except BaseException as error:
                status['state'] = 'failed'
                status['stages'][name].update(state='failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
                atomic_json(status_path, status)
                raise
            finally:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Repeat the synthetic memory check on every allocation; devices may differ.
        stage('memory-check', lambda: atomic_json(run/'memory-check.json', smoke(config, device)),
              ['memory-check.json'], always=True)
        def prepare_data():
            if prepared_from is not None:
                import_prepared(config, run, prepared_from)
            prepare(config, run)
        stage('prepare', prepare_data, ['dataset.json','config.json'])
        stage('real-data-preflight', lambda: real_data_preflight(config, run, device),
              ['real-data-preflight.json'], always=True)
        stage('autoencoder', lambda: train_ae(config, run, device, resume=True),
              ['autoencoder/last.pt','autoencoder/best.pt'])
        stage('validation-reconstruction', lambda: reconstruct_volume(config, run, device, 'val', 0, resume=True),
              ['reconstruction-val-0.nc','reconstruction-val-0.json'])
        stage('cache', lambda: cache_latents(config, run, device), ['latents/manifest.json','latents/provenance.json'])
        stage('flow', lambda: train_flow(config, run, device, resume=True), ['flow/last.pt','flow/best.pt'])
        stage('test-reconstruction', lambda: reconstruct_volume(config, run, device, 'test', 0, resume=True),
              ['reconstruction-test-0.nc','reconstruction-test-0.json'])
        for seed in seeds:
            stage(f'sample-{seed}', lambda seed=seed: generate(config, run, device, seed, resume=True),
                  [f'sample-{seed}.nc', f'sample-{seed}.json'])
        report_files = ['evaluation/metrics.json','evaluation/report.html','evaluation/summary.csv','evaluation/spectra.csv']
        report_files += [f'evaluation/{name}.{suffix}' for name in
                         ('energy-spectrum','kinetic-energy','velocity-pdfs','divergence') for suffix in ('png','pdf')]
        stage('test-evaluation', lambda: evaluate_experiment(config, run, device, seeds), report_files)
        status.update(state='complete', active_stage=None, finished_utc=now())
        atomic_json(status_path, status)
        print('Experiment complete: '+str(run/'evaluation'/'report.html'), flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent.parent/'configs/dns384.json'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--seed-base', type=int, default=50000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepared-from', type=Path, help='Import only verified dataset preparation from an older run')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    experiment(load_config(args.config), args.run, device_for(args.device), args.samples, args.seed_base, args.prepared_from)


if __name__ == '__main__':
    main()
