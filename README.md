# 3D LapFlow for DNS velocity snapshots

Standalone, memory-conscious **adaptation** of [LapFlow](https://arxiv.org/abs/2602.19461).
It generates a complete **384³ three-component velocity volume**, unconditionally.
It does not forecast physical time. No code or artifacts depend on `fluid-vit`.

This is an independent implementation of the multiscale method, **not an exact
port or reproduction of the paper's model**. The pretrained image VAE is replaced
by a small deterministic 3D autoencoder. Width/depth, normalization and numerical
sampling settings are also adapted. See [the architecture and deviations](docs/architecture.md)
before interpreting experimental results.

## Conservative initial configuration

- Original NetCDF fields, `u,v,w`; pressure excluded.
- Autoencoder: widths 8/16/24/32, 16× spatial compression, 8 latent channels.
- Autoencoder training: 64³ core crops, 32-voxel context on each side (128³ input).
- Full-volume encode/decode: aligned periodic halo tiles, retaining only cores.
- Global latent volume: 24³; pyramid 6³/12³/24³, with 2³ patches: 1,971 tokens.
- Transformer: width 192, 6 blocks, 3 heads, three scale-specific branches.
- Query chunks of 256, no dense attention mask, block checkpointing, CUDA BF16.
- Batch size 1. Separate autoencoder and flow training; frozen latents cached on disk.
- Initial training files 131–237 (107 files), validation 451–490, test 492–531.
  Metadata determines snapshot counts; duplicate boundary times are removed.
- Initial budgets: 10,000 autoencoder updates and 20,000 flow updates. These are
  starting settings, not measured convergence or runtime estimates.

These memory settings sacrifice capacity/compression fidelity. Full output
resolution alone does not establish DNS-level accuracy. GPU peak memory is
**unmeasured until the GPU smoke job runs**. The first implementation is single
GPU; it does not split one volume across GPUs or provide distributed training.

## Environment and checks

On Perlmutter:

```bash
cd /global/u1/s/savu/projects/laplace-fm
module load pytorch
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest tests.test_core tests.test_full_stack -v
python -B -m lapflow.train estimate
```

Alternatively install this package in your own Python >=3.10 environment with
`pip install -e .` (PyTorch >=2.5, NumPy, netCDF4, Matplotlib). Installation is not required
when running from the project directory.

## Full-stack experiment (one submission)

```bash
sbatch jobs/full-stack.sbatch
```

This runs the memory check, data preparation, autoencoder training/reconstruction,
latent caching, LapFlow training, eight generated volumes, and held-out evaluation
sequentially on one GPU. It evaluates transport loss and physical statistics on
all test snapshots and writes a self-contained HTML report plus PNG/PDF plots.
The default run is `/pscratch/sd/s/savu/laplace-fm/full-stack-v2`.

Resubmitting the same command resumes unfinished work. The nine-hour allocation
is not a measured completion estimate. See [full-stack stages, metrics and restart
behavior](docs/full-stack.md). Nothing has been submitted automatically.

## Individual stages (optional)

Inspect `configs/dns384.json` first. Choose a fresh scratch run directory for a
changed configuration. Large data, checkpoints and generated files stay outside
this source directory.

```bash
# Short synthetic forward/backward/optimizer check, with peak GPU memory report.
sbatch jobs/memory-smoke.sbatch

# Submit each subsequent stage only after its predecessor finishes successfully.
# All stages use the same default scratch run directory.
sbatch --export=ALL,STAGE=prepare jobs/train-stage.sbatch
sbatch --export=ALL,STAGE=autoencoder jobs/train-stage.sbatch
sbatch --export=ALL,STAGE=reconstruct jobs/train-stage.sbatch
# Inspect reconstruction-val-0.json and .nc before trusting the compression.
sbatch --export=ALL,STAGE=cache jobs/train-stage.sbatch
sbatch --export=ALL,STAGE=flow jobs/train-stage.sbatch
sbatch --export=ALL,STAGE=sample jobs/train-stage.sbatch
```

These are submission instructions, **not a script to paste all at once**: there
are no automatic dependencies between these jobs. Nothing is submitted by the
Python package. `RUN_DIR` and `CONFIG` can be supplied through `--export`; defaults
are `/pscratch/sd/s/savu/laplace-fm/dns384-v1` and `configs/dns384.json`.
`RESUME=1` resumes `last.pt`. Training saves a recovery checkpoint after the first update, at completed-step
boundaries after 60 seconds, and before every scheduled validation (250 updates).
Interruptions can lose work since the latest checkpoint. The script's nine hours is an allocation
ceiling, not a completion estimate. Caching resumes completed latent files.

For an existing allocation, equivalent commands are:

```bash
python -m lapflow.train autoencoder --run /path/to/run --device cuda:0
python -m lapflow.train flow --run /path/to/run --device cuda:0 --resume
python -m lapflow.train sample --run /path/to/run --device cuda:0 --seed 123
```

`prepare` fits velocity normalization on **training-only sampled crops**.
`cache` computes per-channel latent normalization on **all training latents**.
`reconstruct` evaluates the first validation volume by default, with physical and
normalized MSE, relative squared error and kinetic energy; `--split test --index N`
selects another held-out volume. It writes a reconstruction, not a generated future.
`sample` writes `sample-SEED.nc`, restored to the input field's numerical units,
with original spatial coordinates and provenance. Existing outputs are not overwritten.

## Artifacts

- `dataset.json`, `config.json`: source metadata, splits, coordinates, normalization.
- `autoencoder/{best,last}.pt`: reconstruction model and resumable optimizer state.
- `latents/`: frozen full-volume latents, normalization, source/checkpoint provenance.
- `flow/{best,last}.pt`: generative model and resumable optimizer state.
- `*/metrics.jsonl`: training/validation losses and GPU allocation peaks.
- `reconstruction-*.{nc,json}` and `sample-*.{nc,json}`: volume outputs and metadata.

Only load trusted checkpoints. Original DNS files are opened read-only.

### Replacement for failed job 59690040

```bash
sbatch jobs/full-stack-repaired.sbatch
```

Uses a new `full-stack-v2` scratch run and imports only the verified dataset
preparation from `full-stack-v1`. The failed run remains intact. The repaired
reader uses contiguous periodic blocks, and full-volume encoding reads each
snapshot once onto CPU before GPU tile processing. A real-data preflight exercises
wrapped crops, training, validation and full-volume encoding. See
[full-stack documentation](docs/full-stack.md) for timings, recovery and outputs.
