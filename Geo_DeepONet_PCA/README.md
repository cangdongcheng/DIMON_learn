# Geo-DeepONet-PCA

Geometry-conditioned surrogate for the phase-aligned V_m decoder. The network
predicts six intermediate values at every canonical mesh node:

```text
[activation time, PCA coefficient 1, ..., PCA coefficient 5]
```

The fixed decoder reconstructs the waveform as

```text
aligned waveform = node lookup template + residual mean + PCA coefficients @ modes
V_m(x,t) = aligned waveform evaluated at t + reference_AT - predicted_AT
```

Training is supervised in waveform space. Predicted features pass through the
fixed differentiable decoder, and the loss is computed against ground-truth
V_m—not against the feature targets:

```text
predicted [AT, PCA coefficients]
             -> differentiable phase decoder
             -> reconstructed V_m at all 601 times

loss = MSE((reconstructed V_m - ground-truth V_m) / training V_m std)
```

Dividing by the train-only V_m standard deviation makes the loss numerically
well-scaled but does not change its optimum.

The geometry branch takes the 60 PCA geometry parameters and the trunk takes
the four Cobiveco coordinates.  Both are width-200, depth-4 Tanh MLPs, matching
`../Geo_DeepONet`; six learned heads reduce their shared multiplicative
interaction to the six outputs.

## Current result (2026-08-12)

Checkpoint:
`CheckPts/geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep.pt`.
On the fixed 25-heart test split:

| metric | mean +/- std |
|---|---:|
| decoded V_m Rel L2 | 0.13385 +/- 0.01898 |
| decoded V_m MAE | **1.887 +/- 0.380 mV** |
| direct predicted AT MAE | **4.504 +/- 1.066 ms** |
| AT extracted from decoded V_m | 4.533 +/- 1.045 ms |
| decoded max-dV/dt fraction | **0.330 +/- 0.006** |

For context, the held-out oracle decoder at K=5 gives 0.080 mV V_m MAE and an
upstroke fraction of 0.822. The learned model therefore has ample decoder
capacity but still smooths the depolarizing upstroke substantially. Its global
V_m MAE is nevertheless much better than the clean f121 Geo_DONet benchmark
(4.73 mV).

The individual PCA-feature MAEs are very large in this waveform-loss run. That
is expected to be possible—not necessarily desirable—because direct feature
loss is disabled: different AT/coefficient combinations can compensate for one
another after decoding. Treat the predicted features as latent decoder controls
and judge the model using decoded V_m, AT, and dV/dt.

The old width-200/depth-4 AT-only `../Geo_DeepONet` currently gives
7.28 +/- 1.80 ms test AT MAE, versus 4.504 +/- 1.066 ms here. This does not yet
isolate the benefit of waveform supervision. Both use Adam at 5e-4, but the old
default uses all 95 training hearts in one update per epoch; this run uses
batches of eight (about 12 updates per epoch), sampled spatial nodes, and a
learned six-output head.

## 1. Prepare compact targets once

The raw f601 archive is 11 GB compressed (~15 GB in memory).  Run this on a CPU
node with at least 32 GB memory:

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/DIMON_learn/Geo_DeepONet_PCA
python -u prepare_data.py
```

This writes:

```text
/home/svu/e1032484/scratch/geo_deeponet_pca_f601_k5.npz
```

Targets use the train-only decoder already produced by the oracle experiment:

```text
/home/svu/e1032484/scratch/pca_phase_aligned_basis_f601.npz
```

## 2. Train

Submit from this folder:

```bash
qsub train.pbs
```

Or on an interactive GPU node:

```bash
source ~/load_dimon_env.sh
python -u main.py --device cuda
```

The default run uses the same 95/5/25 heart split, width 200, depth 4, and
learning rate 5e-4. A full decoded tensor for 95 hearts x 50,797 nodes x 601
times would exceed the A40's 48 GB after interpolation tensors and gradients
are included. Training therefore samples spatial nodes while retaining every
time point:

```text
8 hearts per optimizer step
2,048 newly sampled mesh nodes per step
601/601 time points for every sampled node
```

New nodes are drawn for every heart batch and epoch. Validation always uses the
same 4,096 nodes across all five validation hearts, making checkpoint selection
stable. Feature AT/PCA MSE values are logged only as diagnostics. By default,
`--feature-loss-weight 0`, so they exert no training pressure. A nonzero value
can add the former feature loss as an auxiliary objective, but that is not the
primary experiment requested here.

On an interactive A40 node:

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/DIMON_learn/Geo_DeepONet_PCA
python -u main.py \
  --device cuda \
  --epochs 5000 \
  --batch-size 8 \
  --nodes-per-step 2048 \
  --val-nodes 4096 \
  --n-components 5 \
  --feature-loss-weight 0 \
  --lr 5e-4 \
  --patience 1000
```

This training path loads the raw f601 V_m archive into CPU memory (~15 GB) but
moves only the current sampled traces to the GPU. Request at least 64 GB of node
RAM. If GPU memory is insufficient, first reduce `--batch-size` to 4, then
reduce `--nodes-per-step` to 1024.

Optional encoder transfer from an old AT-only checkpoint:

```bash
python -u main.py --device cuda --init-at-checkpoint /path/to/old_AT_checkpoint.pt
```

This copies compatible geometry/trunk weights and initializes the new AT head
as the legacy dot product.  The five coefficient heads remain new.

## 3. Evaluate and reconstruct V_m

```bash
python -u main.py --test-model --device cuda \
  --model-path CheckPts/geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep.pt
```

Evaluation reports direct AT/PCA feature errors, reconstructs each held-out V_m
field one heart at a time, and reports V_m Rel-L2/MAE, decoded AT MAE, and
upstroke-slope retention.  It loads the raw V_m archive for these metrics.  For
a quick feature-only evaluation:

```bash
python -u main.py --test-model --device cuda --skip-vm-eval \
  --model-path CheckPts/geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep.pt
```

Outputs are placed in `Predictions/<checkpoint stem>/Test/`.

## 4. Leakage-safe five-fold cross-validation

The PCA decoder itself must be refitted inside every fold. Reusing
`pca_phase_aligned_basis_f601.npz` would leak test-heart waveforms into some
folds because that basis was built for the original fixed split.

`main_cv.py` uses the same seed-42 fold layout as Geo_DONet and Geo_MLP:

```text
five disjoint test folds of 25 hearts
95 fitting + 5 validation hearts per fold
fold-specific node template, PCA modes, feature scales, and V_m scale
5,000 complete epochs with no early stopping
best validation checkpoint retained for test evaluation
```

Waveforms are aligned once to a fixed 80 ms reference. Shared temporal
sufficient statistics are then recombined to construct each fold's PCA basis
using only its 95 fitting hearts. The fixed reference is merely the phase-axis
origin and contains no fitted waveform information.

Submit from this directory:

```bash
qsub cv.pbs
```

The PBS requests one GPU, 16 CPUs, 96 GB RAM, and four hours. Five training
runs may fit within three hours, but the job also constructs five train-only
decoders and their shared temporal sufficient statistics, so the extra hour is
a safety margin. The raw and aligned f601 arrays coexist during the run, making
the larger host-memory request intentional. Outputs are written to:

```text
CV_5fold_5000ep_w200_d4_n2048_f601_vmloss/
```

Each fold contains its leakage-safe `basis.npz`, best `model.pt`, loss curve,
and test metrics. The top level contains `summary.txt`,
`per_case_metrics.csv`, `cv_results.npz`, and `cv_summary.png`.
