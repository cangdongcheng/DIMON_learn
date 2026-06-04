# Geo_DONet_overfit — single-case capacity test for the V_m upshoot

The clean Geo-DONet (`../Geo_DONet/`) reproduces the bulk V_m field but **smears the
sharp ~1–2 ms depolarization upstroke**. Suspected cause: spectral bias — the smooth
Tanh trunk (a coordinate-MLP) can't represent the sharp space-time wavefront.

This experiment isolates **capacity** from **generalization** by overfitting the
network on a single heart (case 0). With one geometry the branch `branch(theta_0)`
is a constant vector `b`, so

    V_m(x, t) = <b, trunk(x, t)>

collapses to a coordinate-MLP (the trunk) with a learned linear head. Driving the
train MSE toward zero on one case measures exactly what the trunk can represent.

- **Fits the upshoot** → capacity is there; the deployed failure is generalization
  (data / conditioning / loss), not the trunk.
- **Still smears it** → spectral bias confirmed. At f601 (1 ms grid) the upstroke is
  temporally resolved, so a residual smear is the *network*, not the sampling — which
  motivates an activation change (SIREN / Fourier features) as the next experiment.

It imports the **actual** `opnn.py` / `utils.py` from `../Geo_DONet/`, so it tests the
same architecture, not a copy.

## Run (compute node only — never the login node)

The script forwards the full grid every epoch, so it must run on a GPU compute node.
One self-contained PBS per resolution (each strides the same `geo_donet_data_f601.npz`
on load); everything is set inside the file — just submit it. Runs cap at 5000 epochs
with early stopping (patience 500 on train MSE), so they usually halt sooner.

```bash
qsub overfit_f121.pbs   # 5 ms sampling (upshoot < 1 frame; deployment grid), ~3 h
qsub overfit_f301.pbs   # 2 ms sampling, grad-checkpointed, ~7 h
qsub overfit_f601.pbs   # 1 ms sampling (upshoot fully resolved; clean capacity read), 12 h
```

Or interactively (e.g. f121, the short one): `python overfit.py --frame-step 5`.
Quick smoke: `python overfit.py --frame-step 5 --epochs 50`.

Outputs land in `Predictions/case0_f{121,301,601}_w300_d4/`.

## Reading the result

| file | what it shows |
|---|---|
| `loss.png` / `loss.txt` | train MSE vs epoch. Plateau above the upshoot error = capacity-limited. |
| `frame_error.png` | per-frame Rel L2 over time. A spike at the depolarization frames = the upshoot is the failure mode. |
| `traces.png` | **the money plot** — GT vs predicted V_m(t) for nodes spanning early→late activation. Does the predicted curve rise as steeply as GT? |
| `at_scatter.png` | predicted vs true activation time; the title quotes AT MAE (ms). |
| `upstroke.png` | max dV/dt per node, GT vs pred. Points below the diagonal = flattened upstroke. |
| `overfit_log.txt` | one-screen numeric summary (final MSE, overall Rel L2, AT MAE, % of GT steepness retained). |

Compare across f121 / f301 / f601: if the smear shrinks as the grid gets finer, sampling
was the limit; if it persists at f601 (1 ms, upshoot fully resolved), the Tanh trunk is
the limit.
