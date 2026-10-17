# One sequential training and testing job

From the project directory:

```bash
sbatch jobs/full-stack.sbatch
```

This requests **one GPU for the process, with a nine-hour limit**. The regular
GPU queue can allocate/bill the whole four-GPU node; this code uses one GPU. Every stage runs sequentially
inside that allocation; there are no child job submissions or manual stage handoffs.
It retains the conservative model/data settings in `configs/dns384.json`. Neither
training time nor total completion time has been measured on the GPU. In particular,
the full experiment can still require multiple allocations; completion within
nine hours is not established.

## Execution order

1. Synthetic forward/backward/optimizer memory check for both configured models.
2. Source metadata discovery and training-only normalization preparation.
3. Real-data preflight: three crop optimizer steps (including three-axis wrapping),
   validation, and one full-volume encoding with disposable weights. Reset the RNG
   before training so this does not change initialization or training data selection.
4. Autoencoder training (10,000 updates), with validation checkpoint selection.
5. Full reconstruction of the first validation snapshot.
6. Frozen encoding/cache of train, validation and test volumes.
7. LapFlow training (20,000 updates), with validation checkpoint selection.
8. Full reconstruction of the first test snapshot using the selected autoencoder.
9. Eight unconditional generated volumes, seeds 50000–50007, saved as NetCDF.
10. Test transport loss on **all test latents**; physical/statistical evaluation of
   **all test DNS snapshots** and all eight generated volumes; plots and report.

Only the selected training split fits weights and normalization; validation selects
checkpoints. Test scores never select weights or sampling seeds. Reconstruction
MSE is measured against the corresponding original DNS snapshot. Generated samples
are unpaired: there is no generated-versus-arbitrary-DNS pointwise MSE.

There is no automatic reconstruction-accuracy gate. Poor but finite reconstructions
are recorded and the experiment continues so the full report is available. This
job tests the baseline; it does not certify physical validity.

## Locations and options

Default run directory:

```text
/pscratch/sd/s/savu/laplace-fm/full-stack-v2
```

Submission logs are `lapflow-full-JOBID.out` and `.err` in the submission directory.
The Python process prints `[start]`, `[done]`, and `[skip]` stage messages.
`pipeline-status.json` in the run directory records the active/completed stage,
errors, durations, source/configuration fingerprints, and artifact metadata.
`memory-check.json` records measured GPU peaks from the synthetic check.

A different experiment directory or sample count can be requested explicitly:

```bash
sbatch --export=ALL,RUN_DIR=/pscratch/sd/s/savu/laplace-fm/full-stack-custom,SAMPLES=16 jobs/full-stack.sbatch
```

`CONFIG` and `SEED_BASE` are also supported. The script defaults to one GPU and
does not implement distributed training. Increasing its GPU request alone will
not distribute the workload.

## Restarting

Resubmit the **same command**, with the same run directory, configuration, source
code and sample-count/seed plan. A filesystem lock prevents simultaneous full-stack
jobs from writing the same run directory. Completed stage artifacts are verified
and skipped, except the inexpensive synthetic memory check, which is repeated on
each allocation. The real-data preflight is also repeated. Training resumes from
`last.pt`, saved after the first update, at completed-step boundaries once 60 seconds
have elapsed, and before scheduled validation. A slow or stuck individual step can
exceed that interval; saving is not an asynchronous timer. Resume validates the
recovered model before continuing, including recovery at the final update. This
can add a validation selection point after an interruption. Cached latent files and per-volume evaluation results are
reused. Atomic NetCDF outputs have embedded provenance, allowing recovery of a
missing JSON sidecar after an interrupted write sequence.

The job does not requeue or resubmit itself. A scheduler kill can leave the active
stage marked `running`; the next invocation retries that stage. An ordinary Python
exception is recorded as `failed`. Changed configuration/source/evaluation plans
require a fresh run directory. Large artifact checks use size/mtime plus embedded
provenance, not complete file-content hashes. Small metadata files are hashed.

## Evaluation definitions

- **Test flow loss:** fixed-seed Monte Carlo transport-vector MSE for each test
  latent and each active scale count. Average equally over snapshots and active
  counts. This is neither field reconstruction MSE nor a physical forecast error.
- **Kinetic energy:** `K = 0.5 * mean(u² + v² + w²)`, including the mean velocity.
  Report distributions across volumes and relative bias of ensemble means.
- **Energy spectrum:** CPU real FFT with `norm='forward'`, Hermitian weights on
  the omitted half spectrum, then shell sums of `0.5 * sum_component |u_hat|²`.
  Angular wavenumbers use coordinate spacing and periodic length `N*abs(dx)`.
  Bins are rounded radial wavenumber / smallest fundamental spacing. The sum
  over all shells equals K; a Parseval discrepancy is reported for verification.
  The plot omits k=0; the CSV retains it. Fourier-cube corner shells are retained.
  This is energy **per shell**, not divided by shell width or shell population.
  Relative L1 discrepancy excludes k=0.
- **Velocity PDFs:** each component uses fixed training mean/std and 120 bins
  on [-6,6]. Density divides by *all* values, not just the in-range count. Fractions
  outside the plotted range are reported separately. Component PDF L1 distances
  are integrated over this fixed range.
- **Divergence:** periodic spectral differentiation, with each even-grid Nyquist
  derivative set to zero. Report RMS divergence and RMS divergence divided by
  the RMS full velocity-gradient tensor. The latter is dimensionless; the raw
  value has velocity / coordinate units. These are diagnostics, not constraints.
- **Autoencoder reconstruction:** full-volume physical and normalized MSE,
  relative squared L2 error and kinetic energy for one validation and one test
  snapshot. This is a small reconstruction spot check, not all-test reconstruction.

Spatial FFTs run on CPU, one volume at a time. The full input volume and FFT work
arrays consume host memory but do not occupy GPU memory. Generation and training
retain the established GPU tiling/latent strategy.

Eight generated volumes are a preliminary sample. The test DNS contains correlated
physical timesteps, so its count does not imply that many independent realizations.
The report does not attach confidence intervals or claim statistical convergence,
Navier–Stokes consistency, or DNS-quality generation.

## Report artifacts

Inside the run directory:

- `evaluation/report.html`: self-contained, with embedded figures; copy it alone
  to a laptop for viewing/presenting.
- `evaluation/energy-spectrum.{png,pdf}`
- `evaluation/kinetic-energy.{png,pdf}`
- `evaluation/velocity-pdfs.{png,pdf}`
- `evaluation/divergence.{png,pdf}`
- `evaluation/metrics.json`: full metrics, per-group statistics, source/seed records.
- `evaluation/summary.csv`, `evaluation/spectra.csv`: tabular summaries.
- `evaluation/per-volume/`: resumable DNS, generated-volume and test-loss results.
- `sample-SEED.nc` and `.json`: generated 384³ velocity volumes and provenance.
- `reconstruction-{val,test}-0.nc` and `.json`: reference reconstruction checks.
- `autoencoder/{best,last}.pt`, `flow/{best,last}.pt`: selected and resumable weights.

## Checks

```bash
module load pytorch
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/laplace-fm-matplotlib-tests \
python -B -m unittest tests.test_core tests.test_full_stack -v
bash -n jobs/full-stack.sbatch
```

The tests execute the complete pipeline on tiny synthetic NetCDF data, simulate an
interruption after writing a volume, recover its metadata, resume all remaining
stages, verify plots/report creation, and reject changed artifacts/plans. Analytic
constant, solenoidal-shear and compressive waves test energy, spectra, Parseval,
divergence, PDF normalization and coordinate handling. No real DNS training or
Slurm submission is performed by these tests.

## Repair of job 59690040 (October 12, 2026)

Submit from `/global/u1/s/savu/projects/laplace-fm`:

```bash
sbatch jobs/full-stack-repaired.sbatch
```

Defaults: `RUN_DIR=/pscratch/sd/s/savu/laplace-fm/full-stack-v2` and
`PREPARED_FROM=/pscratch/sd/s/savu/laplace-fm/full-stack-v1`. This imports only
`dataset.json` and `config.json` after configuration, preparation completion,
artifact fingerprint and original-file size/mtime checks. `prepared-import.json`
records the old source plan and dataset hash. No old training/status files are
copied; the new pipeline records the repaired source hash. Resubmit the same
unchanged script to continue after a time limit. It does not submit itself.

Scientific settings are unchanged: 384³ velocity volumes; 964/361/361 deduplicated
train/validation/test snapshots; 10,000 autoencoder updates; 20,000 flow updates;
the same architecture, losses and eight generated samples.

The reader now splits periodic windows into contiguous blocks (at most 24 NetCDF
reads for a three-channel window no larger than the grid), rather than orthogonal
index arrays that caused millions of reads. Full-volume encoding holds one raw
648 MiB velocity snapshot on CPU, then extracts and normalizes tiles in memory;
GPU processing remains tiled. This adds bounded host memory, not a full DNS
volume on GPU. Invalid encoded latents are rejected before publishing cache files.

`autoencoder/progress.jsonl` and `flow/progress.jsonl` log loss and data/step timing
for the first three updates, every ten updates, and checkpoint updates. Validation
metrics remain in `metrics.jsonl`; `last.pt` is for recovery, `best.pt` for validation
selection. Early recovery checkpoints need not yet have a corresponding `best.pt`.
`real-data-preflight.json` records real crop training/validation and full-snapshot
encoding timings, with rough AE/cache runtime estimates. These estimates exclude
flow training, generation, final evaluation and filesystem variability.

CPU verification: all 19 tests passed, including bitwise-identical resumed model
weights after interruption during validation, bounded contiguous-read counts,
periodic indexing across all boundary combinations, and safe preparation import.
The formerly problematic crop from fluid_229.nc/index 1, origin (-32,320,320),
size 128³, loaded in 1.39 seconds and matched a full-volume reference exactly.
One full snapshot loaded in 1.72 seconds. These are individual measurements,
not sustained-throughput guarantees. No repaired full-scale GPU training was
launched as part of this fix; the job performs that preflight on its allocation.
