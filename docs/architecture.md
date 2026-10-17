# Model specification and scientific deviations

## Scope and source

The task is unconditional generation of one joint 384×384×384 velocity volume,
with three channels u,v,w. The output has no physical forecast lead. There is no
conditioning history, rollout, pushforward training, pressure channel, PDE loss,
spectral loss, or divergence projection. Periodicity is an explicit assumption
inherited from the HIT problem, not an assertion inferred from NetCDF attributes.

References inspected on 2026-10-11:

- [LapFlow paper](https://arxiv.org/html/2602.19461v1), especially Eqs. 3–7 and Algorithms 1–2.
- [Official model](https://github.com/sjtuytc/ICLR26_LapFlow_Official/blob/main/models/MultiScaleDiT.py).
- [Official noise/paths](https://github.com/sjtuytc/ICLR26_LapFlow_Official/blob/main/sampler/ours_sample.py)
  and `sampler/edify_image_sample.py` in the same repository.

This project is independently written; no upstream model source is vendored.
Links identify the inspected sources; upstream `main` may change. This is a
small 3D source-informed adaptation, not an exact paper reproduction.

## 1. Physical fields and normalization

Each original `fluid_ID.nc` contains ten recorded states at a 384³ spatial grid.
The loader checks dimensions and coordinate equality and extracts small periodic
windows directly from the NetCDF arrays. It never loads an entire ten-state
file into GPU memory. Shared exact timestamp boundaries within a split are
counted once. Overlapping file ranges and identical timestamps across splits
are rejected. Adjacent temporal samples are still correlated; these splits do
not constitute independent realizations or Reynolds-number generalization.

Initial file ranges are the previous small-experiment convention: train 131–237,
validation 451–490, test 492–531. This retains 107 training files, not all 401.
It is a transparent starting configuration, not a scientifically optimized split.

For each velocity channel, mean and standard deviation are estimated from 128
seeded random training crops (64³ each), not the validation/test data. This is a
memory/I/O compromise: the statistics are estimates, not exact global moments.
No image clipping, RGB conversion, or image-pretrained intensity scaling is used.

## 2. The low-memory autoencoder

This is the largest departure from LapFlow's image pipeline. We train a small
**deterministic convolutional autoencoder**, not a pretrained image VAE. There is
no posterior sampling, KL term, discriminator, or image perceptual loss.

Encoder, written as input channels → output channels:

1. Conv3D 3→8, kernel 3, stride 2, padding 1; SiLU.
2. Conv3D 8→16, kernel 3, stride 2, padding 1; SiLU.
3. Conv3D 16→24, kernel 3, stride 2, padding 1; SiLU.
4. Conv3D 24→32, kernel 3, stride 2, padding 1; SiLU.
5. Conv3D 32→8, kernel 1, stride 1.

Thus 384³ becomes 24³ with eight latent features per point. This is a 1,536-fold
reduction in scalar count relative to the three-channel physical volume. It is
aggressive; unresolved fluctuations can be lost despite full-resolution output.

Decoder: 1×1×1 convolution 8→32 + SiLU, then transposed convolutions
32→24→16→8→3, each kernel 4, stride 2, padding 1. Intermediate outputs use SiLU;
the physical output is unbounded. There is no spatial normalization or attention
inside the autoencoder. This preserves a finite local receptive field and makes
exact halo tiling possible. It differs substantially from the original image VAE.

Training uses 64³ target cores with 32 context voxels on each side (128³ inputs).
Crop origins are aligned to the 16-voxel compression lattice. The loss is mean
squared error on the central normalized velocity core. The first convolution
strides immediately, avoiding a large full-resolution hidden feature tensor.

For encoding a whole field, encode each tile with its periodic halo and retain
only the central 4³ latent core. For decoding, take context from the same joint
latent volume and retain the central physical core. There is no blending and no
independently generated tile noise. Tests compare this against periodic whole-
volume convolution, including tile edges and the deployed training core.

The GPU holds one tile at a time. The complete decoded output is assembled in
CPU memory (~648 MiB float32 for velocity alone). CPU metrics/denormalization
require additional copies. Overlapping halo extraction adds CPU work; encoding all original
volumes is not a trivial preprocessing operation or guaranteed short job.

## 3. Laplacian hierarchy in latent space

After freezing the best validation-MSE autoencoder, cache each complete latent
volume. Compute a second per-channel normalization using all training latents.
Keep these statistics and the encoder checkpoint hash with the cache.

For a normalized latent Z at 24³:

- C = Down(Down(Z)), at 6³.
- R_mid = Down(Z) − Up(C), at 12³.
- R_fine = Z − Up(Down(Z)), at 24³.
- Z = Up(Up(C)) + Up(R_mid) + R_fine.

Down is 2³ average pooling and Up is nearest-neighbor expansion. These operations
are exactly reconstructive when residuals are retained. The latent features are
learned; these bands are not literal physical Fourier shells or eddy sizes.
All code uses coarse-first ordering, opposite to the paper's finest-first indices.

## 4. Tokens and attention

Each level has a distinct Conv3D patch projection (kernel=stride=2). The three
levels contribute 27, 216, and 1,728 tokens respectively: **1,971 total**. Each token
has 192 features, plus fixed 3D sinusoidal position encoding. Sampling time is
encoded with 64 sinusoidal features and a learned two-layer MLP.

Six transformer blocks process these tokens. Each block has separate modulation,
Q/K/V projections, attention output projections and MLPs for each scale. Three
heads have 64 features each. MLP hidden width is 768. Adaptive modulation gates
start at zero; final output heads start at zero. Label conditioning is omitted
because the experiment is unconditional. Learned scale-specific modulation and
heads, plus positional encoding, are not the image network's weights.

For attention, a coarse query sees all coarse keys; a middle query sees coarse
and middle keys; a fine query sees every key. Compute QK^T, multiply by
0.25/sqrt(head_width), apply softmax, and combine V. The 0.25 factor preserves the
public source's attention scale; it is not the usual bare 1/sqrt(head_width).

The implementation constructs those allowed K/V prefixes directly, rather than
allocating an N×N causal mask for each block. Each query set is evaluated in chunks
of at most 256. Dense-reference output and gradient tests verify equivalence.
SDPA may select a fused or math implementation depending on hardware. We make no
claim that a specific kernel ran. At batch one the largest float32 score chunk is
about 5.8 MiB; this is **not peak training memory**. Gradients, model states and
other activations also consume memory. Block-level non-reentrant checkpointing
recomputes activations in backward. It is implemented here and tested.

All patch outputs are rearranged back to 3D arrays with eight channels; each scale
has its own correctly indexed output head. No slice stacking is involved.

## 5. Flow objective and generation

Starts, coarse-first, are a=[0,1/3,2/3]. Sample an active level count uniformly
from {1,2,3}, then sample continuous generation time t uniformly from the start
of the finest active level to 1. For each active level i:

    x_i(t) = ((t-a_i)/(1-a_i))*z_i + (1-t)*epsilon_i
    target_i = z_i/(1-a_i) - epsilon_i
    L = sum_i mean((prediction_i - target_i)^2)

Each level has weight one. Mean includes batch, channel and spatial dimensions;
there is no additional division by active scale count. This is transport-vector
MSE, not physical velocity forecast MSE. Training takes one sampled generation
time, not a differentiable rollout of all sampling steps.

**Paper/code discrepancy retained explicitly:** the inspected public three-level
noise helper uses independent unit Gaussian noise at each resolution. The paper
algorithms describe a Laplacian decomposition of one noise field. We follow the
public three-level helper. Two-level configuration follows the public helper's
Laplacian-of-one-noise construction. Training and sampling share the same helper.

Generation integrates coarse states from 0 to 1/3, adds middle noise scaled by
1−1/3, integrates both to 2/3, adds fine noise scaled by 1−2/3, then integrates all
three to 1. Coarser levels remain active. Use fixed-step Heun with 16 steps per
segment: 48 ODE steps and 96 evaluations. This changes the paper's adaptive solver
and trades numerical accuracy for bounded cost. Increasing steps is configurable.

Finally reconstruct the latent pyramid, undo latent normalization, release the
transformer, decode tiled on the accelerator, assemble on CPU and undo velocity
normalization. New random seeds produce different samples. A fixed seed is
repeatable within the same numerical environment, not guaranteed bitwise across
GPU architectures. Neither the ODE time nor its 48 steps is DNS physical time.

## 6. Validation and limitations

Autoencoder best checkpoint: lowest normalized MSE on a fixed small set of seeded
held-out crops. Flow best checkpoint: lowest sampled transport loss on the first
few held-out latents, evaluated with a fixed RNG and each active scale count.
These are inexpensive diagnostics, not comprehensive physical validation.
`reconstruct` evaluates a complete held-out volume and reports physical MSE,
normalized MSE, relative squared error, and total kinetic energy per unit mass
K=0.5*mean(u²+v²+w²). The full-stack job additionally evaluates energy spectra, velocity PDFs and
spectral divergence against all test snapshots; see [metric definitions](full-stack.md).
Higher-order increment statistics are not implemented. Generated volume quality
and conservation remain unestablished until actual experiments are evaluated.

Source metadata, config, velocity/latent statistics, checkpoint hashes, losses,
and RNG/optimizer state are recorded. Original DNS is read-only. Cache provenance
rejects a changed autoencoder/config. Resuming training restores the last saved
optimizer and RNG state; checkpoint cadence defines the maximum lost progress.

This version does **not** implement multi-GPU data, tensor, or context parallelism.
Memory is reduced within one GPU first; ordinary data parallelism would not fix
an oversized sample. CPU tests establish implementation behavior, not GPU fit or
physics. A short GPU smoke script measures both forward/backward and the first
optimizer allocation before any expensive experiment.


### I/O and recovery repair (October 12, 2026)

Periodic NetCDF crops use contiguous source/destination blocks, never wrapped
integer index arrays. Full-volume encoding loads one 648 MiB raw velocity snapshot
on CPU and reads overlapping tiles from memory, avoiding repeated disk halo reads.
Only tiles enter GPU memory. Recovery checkpoints are independent of validation:
first update, completed-step boundaries after 60 seconds, and before validation.
The full-stack job now performs a real-DNS training and full-encoding preflight.
These changes preserve the mathematical model, data split, loss and update counts.
See `full-stack.md` for the replacement job and measured verification.
