# Geo-DeepONet-PCA

Geometry-conditioned feature surrogate for the phase-aligned V_m decoder.
Instead of predicting all 601 values of V_m(x,t), the network predicts six
values at every canonical mesh node:

```text
[activation time, PCA coefficient 1, ..., PCA coefficient 5]
```

The fixed decoder reconstructs the waveform as

```text
aligned waveform = node lookup template + residual mean + PCA coefficients @ modes
V_m(x,t) = aligned waveform evaluated at t + reference_AT - predicted_AT
```

The geometry branch takes the 60 PCA geometry parameters and the trunk takes
the four Cobiveco coordinates.  Both are width-200, depth-4 Tanh MLPs, matching
`../Geo_DeepONet`; six learned heads reduce their shared multiplicative
interaction to the six outputs.

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

Defaults reproduce the old AT-only setup where applicable: 95/5/25 split,
width 200, depth 4, full 95-heart batch, learning rate 5e-4, and 50,000 epochs.
Every target channel is independently standardized using the 95 training
hearts. The loss has two equal groups by default:

```text
loss = 0.5 * MSE(normalized AT)
     + 0.5 * sum_k w_k * MSE(normalized PCA_k)
```

The within-PCA weights `w_k` come from the f601 decoder basis's explained
variance and are renormalized over the selected five modes. Thus AT receives
half of the training pressure, rather than only one-sixth, while the PCA half
emphasizes the waveform modes with the greatest physical contribution. Both
group weights can be overridden with `--at-loss-weight` and
`--pca-loss-weight`. `loss.csv` and `loss.png` record total, AT, and PCA losses
separately.

Optional encoder transfer from an old AT-only checkpoint:

```bash
python -u main.py --device cuda --init-at-checkpoint /path/to/old_AT_checkpoint.pt
```

This copies compatible geometry/trunk weights and initializes the new AT head
as the legacy dot product.  The five coefficient heads remain new.

## 3. Evaluate and reconstruct V_m

```bash
python -u main.py --test-model --device cuda \
  --model-path CheckPts/geodeeponet_pca_k5_w200_d4_50000ep.pt
```

Evaluation reports direct AT/PCA feature errors, reconstructs each held-out V_m
field one heart at a time, and reports V_m Rel-L2/MAE, decoded AT MAE, and
upstroke-slope retention.  It loads the raw V_m archive for these metrics.  For
a quick feature-only evaluation:

```bash
python -u main.py --test-model --device cuda --skip-vm-eval \
  --model-path CheckPts/geodeeponet_pca_k5_w200_d4_50000ep.pt
```

Outputs are placed in `Predictions/<checkpoint stem>/Test/`.
