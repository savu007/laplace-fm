# Understanding our 3D LapFlow baseline

This describes the implemented `configs/dns384.json` model, checked against source on October 12, 2026. It is an independently implemented **3D adaptation of LapFlow**, not the authors’ unchanged implementation. The principal adaptation is a small, deterministic 3D autoencoder trained on our DNS. See [the detailed specification](architecture.md) for upstream references and recorded paper/code differences.

## 1. What mathematical problem are we solving?

A training example is one velocity field **U = (u,v,w)** on a 384³ grid. We want to sample a new field from a distribution resembling the training snapshots. There is no input history, pressure channel, Reynolds-number condition, or physical-time condition. If the data span different physical states, their distribution is learned without telling the model which state to produce.

The complete model has two separately trained parts:

```text
Learn a representation:
DNS velocity → normalize → encoder → 8-channel 24³ latent → decoder → reconstructed velocity

Learn to generate that representation:
Gaussian noise → multiscale latent ODE → generated 24³ latent → frozen decoder → generated velocity
```

The encoder/decoder learns how to represent a field. The transformer learns which representations resemble the dataset. Neither is trained to advance Navier–Stokes in time.

## 2. Compression is the first scientific bottleneck

The encoder applies four 3D convolutions, each halving spatial resolution:

| Operation | Channels | Spatial grid |
|---|---:|---:|
| Input | 3 | 384³ |
| Convolution + SiLU | 8 | 192³ |
| Convolution + SiLU | 16 | 96³ |
| Convolution + SiLU | 24 | 48³ |
| Convolution + SiLU | 32 | 24³ |
| 1³ convolution | 8 | 24³ |

The first four kernels are 3³, stride 2, padding 1. SiLU is a smooth nonlinear activation. The decoder maps 8→32 channels, then uses four learned transposed convolutions (kernel 4³, stride 2) with channels 32→24→16→8→3. Its final output is unbounded. Together these networks have **119,347 parameters**.

Eight latent channels are learned features, not eight physical variables. Scalar count falls from 3×384³ to 8×24³: **1,536-fold compression**. Returning a 384³ array does not mean recovering DNS-resolved information. The decoder must infer fine detail from this restricted representation.

Training minimizes mean squared reconstruction error on normalized velocity. Channel means/stds come from sampled training crops. Thus physical component errors are effectively weighted by inverse variance; this is not an energy or derivative loss. No KL penalty, adversarial loss, stochastic decoder, or physical constraint is used.

To save memory, training supplies a 128³ crop but scores only its central 64³ core. The surrounding 32-voxel halo provides context. During whole-volume encoding, only each tile’s central 4³ latent core is retained. During decoding, overlapping context comes from **one joint latent volume**, and only physical cores are retained. These are not independently generated pieces stitched together. Periodic halos and finite convolutional receptive fields make tiled computation equivalent to the corresponding periodic full-volume operation within numerical tolerance.

The selected autoencoder is then frozen. All snapshot latents are cached, and a second per-channel normalization is fitted using training latents. Flow training cannot update or repair the encoder/decoder. Reconstruction spectra are therefore a crucial first diagnostic: if compression already destroys small-scale energy, generation has a substantial representation problem before its own errors are considered. This is a diagnostic limitation, not a formal bound on every generated statistic.

## 3. What “Laplacian multiscale” means here

Let Z be the normalized 24³ latent. Define D as 2³ average pooling and A as nearest-neighbor upsampling:

```text
C = D(D(Z))                : 6³ coarse field
R = D(Z) − A(C)            : 12³ residual
F = Z − A(D(Z))            : 24³ residual
Z = A(A(C)) + A(R) + F     : exact reconstruction
```

R and F represent what the next coarser level cannot express. This decomposition loses no additional information when all bands are retained. It is not a Laplacian PDE operator, and these learned-feature bands are not physical Fourier shells.

True residuals have zero average within their corresponding 2³ blocks. The network does **not** explicitly project generated residuals onto that constraint; it must learn their distribution.

## 4. Exactly what attention computes

Each band has its own learned 2³ patch projection. Eight latent channels over eight cells become one 192-feature token:

| Band | Token grid | Tokens | Keys accessible to each query |
|---|---:|---:|---|
| C, 6³ | 3³ | 27 | All 27 coarse tokens |
| R, 12³ | 6³ | 216 | All 243 coarse + middle tokens |
| F, 24³ | 12³ | 1,728 | All 1,971 tokens |

There are **six transformer blocks**, width **192**, **three attention heads** of width 64, and a per-token nonlinear MLP with hidden width **768**. Each scale has separate projection, MLP, and modulation weights. Total transformer parameters: **12,342,144**.

For token p, learned linear maps produce a query qₚ, key kₚ, and value vₚ. Within each head:

```text
weight(p,q) = softmax over allowed q [ (0.25 / √64) · qₚᵀk_q ]
message(p)  = Σ_q weight(p,q) · v_q
```

The query describes what information the receiving token seeks; keys determine compatibility; values supply the information being combined. These are learned feature vectors, not physical velocities. The 0.25 factor is explicitly present in our implementation. Attention is global within the allowed scale set: a fine token can receive information from the opposite side of the domain in one block.

The direction is asymmetric: fine detail can depend on large-scale organization; coarse organization cannot inspect finer tokens. Here “causal” means **coarse-to-fine information dependence**, not physical-time causality. It does not enforce a turbulent energy cascade.

Fixed 3D sinusoidal positions distinguish token locations. Generation-time embeddings control feature shifts, scales, and residual gates in each block, adapting its computation to the current noise level. The gated attention message is added to the token, followed by a gated MLP update. Final per-scale heads turn each token into 8×2³ numbers and rearrange them into a latent transport field.

Query chunks of 256 reduce working memory without dropping allowed keys. Gradient checkpointing recomputes blocks during backpropagation. These are execution strategies, not changes to the attention pattern. Periodic data handling does not make the position encoding or attention explicitly periodic or translation invariant.

## 5. What flow matching learns—and what its loss means

Use τ for **generation time**, to distinguish it from DNS time. For clean band zᵢ, Gaussian noise εᵢ, and activation times a = (0, 1/3, 2/3), the training path is:

```text
xᵢ(τ) = [(τ − aᵢ)/(1 − aᵢ)] zᵢ + (1 − τ) εᵢ
bᵢ    = dxᵢ/dτ = zᵢ/(1 − aᵢ) − εᵢ
```

The transformer predicts bᵢ: how the noisy **latent coordinates** should move. This transport vector is not fluid velocity or acceleration. Training uses independent Gaussian noise at the three resolutions; this follows the public-code behavior recorded in our detailed specification, rather than decomposing one noise field into three Laplacian bands.

Each update chooses an active-band count m uniformly from {1,2,3}, samples τ uniformly from [aₘ₋₁,1], and minimizes:

```text
Lflow = Σ over active bands mean[(predicted transport − bᵢ)²]
```

Means are taken separately per band; there is no division by m. Consequently, equal coefficients do not imply equal total influence: coarse appears every update, middle two-thirds, fine one-third; target scales also differ. This loss is neither decoded-field MSE nor spectral/energy error.

Different clean samples can correspond to similar noisy inputs with different target directions. Squared loss fits their conditional-average transport. Integrating that learned transport can still produce diverse fields because trajectories begin from different random noise. A deterministic network therefore does not imply a single generated outcome.

## 6. How one new volume is generated

Start with coarse noise at τ=0. At τ=1/3 introduce middle noise with amplitude 2/3; at τ=2/3 introduce fine noise with amplitude 1/3. Previously active bands keep evolving. Coarse is not frozen when fine detail appears.

The sampler uses Heun’s predictor–corrector method: evaluate transport, predict an endpoint, evaluate again, then advance using the average slope. Sixteen steps per interval gives **48 ODE steps / 96 network evaluations**. These are numerical generation steps, not 48 physical timesteps. Training samples individual path points; it does not backpropagate through this sampler.

At τ=1, reconstruct Z from the three bands, undo latent normalization, decode the joint field in tiles, and undo velocity normalization. The output is one unconditional u,v,w volume.

## 7. What would count as scientific success?

First check reconstruction: DNS → encoder → decoder. Then separately check generated versus held-out distributions using energy spectra, kinetic energy, velocity PDFs, and divergence. This separates compression failure from a failure to learn the latent distribution. Pairwise MSE against an arbitrary DNS snapshot is not a meaningful test of an unconditional sample.

The current setup has no incompressibility projection, Navier–Stokes residual, isotropy constraint, or energy-budget constraint. Even matching all reported statistics would not establish plausible **temporal evolution**, because no evolution is modeled. The immediate claim to test is narrower: whether this compressed, multiscale generator reproduces selected spatial statistics of the training regime.

**Code map:** [configuration](../configs/dns384.json), [encoder/decoder and tiling](../lapflow/autoencoder.py), [attention](../lapflow/model.py), [paths, loss, sampler](../lapflow/flow.py), [training](../lapflow/train.py), [evaluation definitions](full-stack.md).
