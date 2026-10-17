# Ten-minute speaking guide

Deck: [lapflow-10-minute-presentation.tex](lapflow-10-minute-presentation.tex).
The first nine slides are the talk; three technical backup slides are for questions.
Edit the name, laboratory, and date on the title slide. No figures or data files
are needed to compile. The deck also contains hidden Beamer speaker notes.

| Slide | Time | Cumulative end |
|---|---:|---:|
| 1. Title | 0:20 | 0:20 |
| 2. Research question | 0:40 | 1:00 |
| 3. Why a small encoder | 1:20 | 2:20 |
| 4. What encoder/decoder learn | 1:15 | 3:35 |
| 5. Compression bottleneck | 1:15 | 4:50 |
| 6. Laplacian decomposition | 1:20 | 6:10 |
| 7. Coupled transport velocities | 1:20 | 7:30 |
| 8. Progressive generation | 1:00 | 8:30 |
| 9. Evidence and evaluation | 1:30 | 10:00 |

About four minutes explain the representation; about four explain the method and
its motivation. The timings are a rehearsal budget, not a measured speech duration.

## 1. Title

“This is a baseline for generating three-dimensional turbulent velocity snapshots.
I will focus on the representation problem and how LapFlow organizes generation
across scales.”

## 2. Research question

“We want a new joint velocity volume with spatial statistics resembling the DNS
examples. This is unconditional generation: there is no starting snapshot or
physical forecast lead. The method has two learned parts. First, an autoencoder
learns a compact description of a field. Second, a generative model learns the
distribution of those descriptions. We evaluate decoded outputs using kinetic
energy, spectra, PDFs, and divergence. Temporal dynamics are outside this baseline.”

## 3. Why a small encoder

“A velocity volume itself is about 648 MiB in single precision. Hidden feature
volumes can be much larger: 128 channels at the full grid would require 27 GiB for
just one tensor. That is an illustration of scaling, not the memory allocation of
our actual network. Backpropagation and optimizer state increase the requirement.

“We chose a conservative representation after concerns about memory and execution
cost: immediately downsample, use narrow convolutional layers, and process local
crops with context. We train a deterministic 3D autoencoder on our DNS instead of
using an image VAE unchanged. This is a practical starting choice, not evidence that
it is optimal. A 3D VAE is possible, but making the latent probabilistic would not
by itself recover information lost through aggressive compression.”

## 4. What the encoder and decoder learn

“Each encoder convolution mixes the three velocity components and neighboring
values through learned filters, followed by a nonlinear activation. Stride two
halves each spatial dimension. Four layers take the grid from 384 cubed to 24
cubed, while increasing feature channels; a final pointwise projection selects
eight latent features.

“These features have no prescribed physical meaning. The decoder learns how to
expand them using transposed convolutions. It is not an algebraic inverse. Both
networks are optimized together to reconstruct the original normalized velocity.
They are then frozen before flow training. Training uses a 128-cubed crop but
scores only the central 64-cubed core; the halo supplies spatial context.”

## 5. Compression bottleneck

“The representation contains 1,536 times fewer scalar values. Returning a
384-cubed output grid does not mean retaining all DNS-resolved detail. Multiple
fine structures may receive similar latent descriptions, and squared error can
encourage their average. That may smooth fluctuations or suppress high-wavenumber
energy. This is a risk to measure, not an established result from our full run.

“The flow model is restricted to what the frozen decoder can express. We should
therefore evaluate reconstruction separately from generation. Compare an actual
DNS field with its own reconstruction using spectra and energy as well as MSE.
If the representation is poor, replacing the generator alone may not help.
Periodic halo tiling keeps computation local while decoding one joint latent;
it does not certify physical fidelity.”

## 6. Laplacian decomposition

“Now we take the learned latent and apply fixed arithmetic. Pool to 12 cubed and
again to 6 cubed. Store what each coarser approximation misses as a residual.
Upsampling and adding the residuals exactly reconstructs the original latent.

“For a small example, [2,4,8,10] averages to [3,9]. Repeat those averages to obtain
[3,3,9,9], and the residual is [-1,1,-1,1]. Add the residual and you recover the
original. In 3D we average groups of eight cells.

“This decomposition itself loses no additional information. The encoder is where
information may already have been lost. The bands are learned-feature scales,
not physical Fourier shells, and the word Laplacian does not mean solving a PDE.”

## 7. Coupled transport velocities

“The transformer predicts a rate of change for each noisy band in generation
time. Coarse transport sees coarse information; middle sees coarse and middle;
fine sees all three. Separate parameters specialize by scale, while attention
lets fine tokens select coarse context using queries, keys, and values.

“Training adds the active velocity errors and uses one backward pass. A fine
error can improve coarse internal features because fine attention uses their
keys and values. Information travels coarse to fine during prediction; gradients
travel backward during learning. The final coarse output head still receives its
own velocity loss. This organization encourages detail consistent with broad
latent structure, but does not enforce a physical energy cascade.”

## 8. Progressive generation

“At the start only coarse noise evolves. At one-third generation time, middle
residual noise is introduced; at two-thirds, fine residual noise is introduced.
The existing coarse states keep evolving. Heun's method integrates all active
states with a predictor and a corrected update. Finally we add the bands,
denormalize the latent, decode one field, and restore velocity units.

“Pyramid Flow instead changes the resolution of one state and uses corrective
renoising between stages. Our method retains distinct band coordinates. Noise
seeds provide diverse samples even though network evaluation is deterministic.
None of these integration steps is a physical DNS timestep.”

## 9. Evidence and evaluation

“We have verified execution through an actual Perlmutter GPU run at the full
spatial geometry and configured model sizes. It trained, checkpointed,
reconstructed, generated, and made a report. A second launcher invocation verified
and skipped completed stages. That run used three updates per model and one
snapshot per split, so it demonstrates the pipeline rather than learned quality.

“The original long job failed in data access: wrapped index arrays caused millions
of NetCDF library calls. Contiguous block reads repaired that path; early recovery
checkpoints now protect completed updates. Keep this execution issue distinct from
the scientific encoder problem.

“Next, reconstruction must retain useful energy and structure. Then generated
ensembles should agree with held-out DNS in spectra, energy, PDFs, divergence,
and diversity. These diagnostics are not training constraints. The representation
is a major hypothesis to test, and our implementation is a 3D adaptation of the
image method rather than an exact reproduction.”

## If time is short

Keep slides 3--7. Shorten optimization details and the job-failure explanation;
do not skip the distinction between compression and the exactly reconstructive
pyramid. End with the evidence-qualified conclusion on slide 9.

## Compilation

```bash
module load texlive/2024
pdflatex -halt-on-error lapflow-10-minute-presentation.tex
pdflatex -halt-on-error lapflow-10-minute-presentation.tex
```

Run these commands from the `docs` directory, or upload the `.tex` to Overleaf.
The supplied audience PDF hides speaker notes; technical backups follow slide 9.
