# Geo-DONet (SIREN trunk)

A geometry-conditioned DeepONet that maps a heart's PCA geometry code **θ → transmembrane
voltage field Vₘ(x, t)** on a fixed reference mesh — same as the clean [`../Geo_DONet`](../Geo_DONet),
but the trunk's Tanh activations are replaced by **SIREN sinusoids** (Sitzmann et al., NeurIPS
2020) to fit the sharp depolarisation upstroke the smooth Tanh trunk smears. The branch stays
Tanh (the PCA geometry modes are smooth; sine buys nothing there).

```
main.py    orchestration: parse_args, train(), infer(), main()
opnn.py    architecture: GeoDONetSIREN (SineLayer trunk), trunk chunking, checkpoint loading
utils.py   data (load_dataset/Normalizer) + evaluation metrics + ndiff (signed-Laplacian) loss
viz.py     metric tables, traces, 3D scatter SVGs, colorbars, .vtu series
```

This is a **self-contained copy** of the baseline's `utils.py`/`viz.py` (so the folder stands
alone) plus a SIREN `opnn.py`/`main.py`. The pre-rewrite monolithic version is archived under
`_old_monolithic/`.

## Setup

```bash
source ~/load_dimon_env.sh        # torch 2.1.2+cu121 venv; sklearn, meshio
cd Geo_DONet_SIREN
python main.py ...                # run from inside this folder (outputs are cwd-relative)
```

## The three modes

| mode | command | what it does |
|---|---|---|
| **train** (default) | `python main.py` | train + validate; save best-val weights |
| **test** | `python main.py --test-model --model-path CKPT` | predict the **test split** + evaluate vs GT |
| **infer** | `python main.py --infer --model-path CKPT --data-path NPZ` | predict **all** cases; no GT needed (`--eval` to score) |

```bash
# train SIREN baseline (f121 = f601 strided by 5), defaults: omega_0=30, flat lr 1e-4
python main.py

# scheduled LR + the ndiff spatial loss (the active SIREN+ndiff experiment)
python main.py --lr 5e-4 --lr-scheduled --ndiff-weight 1.0

# evaluate on the held-out test split -> per-case tables + mean ± std summary
python main.py --test-model --model-path CheckPts/geodonet_siren_w300_d4_w0_30_5000ep_f121.pt

# re-evaluate an OLD original-opnn SIREN checkpoint (loads transparently, omega_0=30)
python main.py --test-model \
  --model-path CheckPts/model_chkpts_geo_donet_siren_5000ep_w300_w0_30_lrsched_ndiff1_f121.pt \
  --data-path geo_donet_data_f601.npz --frame-step 5

# evaluate + export figures + npz
python main.py --test-model --model-path CKPT --snapshot --color-bar --save
```

## How configuration works

Every run default lives in the **CONFIGURATION block** at the top of `main.py`, and **every
default is exposed as a CLI flag that overrides it**. Run with no flags for the hardcoded config.

Key defaults (SIREN-specific in **bold**):

| constant | default | meaning |
|---|---|---|
| `DATA_FILE` / `FRAME_STEP` | `geo_donet_data_f601.npz` / `5` | default npz + time stride → f121 |
| `WIDTH, DEPTH` | `300, 4` | branch/trunk MLP size |
| **`OMEGA_0`** | **`30`** | **SIREN trunk frequency (lower = smoother)** |
| `EPOCHS, BATCH_SIZE, LR` | `5000, 24, 1e-4` | optimization (SIREN uses a smaller flat LR than Tanh) |
| `GRAD_CHECKPOINT` | `True` | **on by default** — SIREN trunk activations need it even at f121 |
| **`NDIFF_WEIGHT`** | **`0.0`** | **signed-Laplacian neighbour-diff loss weight (0 = MSE only)** |

## SIREN-specific flags (over the baseline)

| flag | default | notes |
|---|---|---|
| `--omega-0` | 30 | trunk frequency bandwidth; stored in the checkpoint config |
| `--ndiff-weight` | 0 | λ on `mean((L(pred−gt))²)`, `L = I − D⁻¹A`; train-only auxiliary loss |
| `--adj-file` | `mesh_adjacency_data_order.npz` | canonical-order mesh adjacency (V_m node order) for ndiff |
| `--grad-checkpoint` | on | `--no-grad-checkpoint` to disable; ~1.3× slower, identical result |

All other flags (`--epochs`, `--lr`, `--lr-scheduled`, `--frame-step`, `--test-model`, `--infer`,
`--cases`, `--snapshot`, `--vtu-out`, `--color-bar`, `--save`, `--n-viz`, `--reference-xyz`,
`--mesh`, `--train-data`, …) behave exactly as in `../Geo_DONet` — see its README.

## Checkpoints

Checkpoints store the **model weights plus a `config` block** (`{model_state_dict, config}`).
Unlike the shape-only GeoDONet checkpoints, the config is needed because `omega_0` is a
forward-time multiplier, not a weight, so it can't be inferred from the state_dict. On load:

- **New** GeoDONetSIREN checkpoints → architecture taken from the saved `config` (incl. `omega_0`).
- **Legacy** original-opnn SIREN checkpoints (`_branch_g`/`_trunk`/scalar `bias`, no config) load
  transparently: keys are remapped, the architecture is inferred from the weight shapes, and
  `omega_0` defaults to **30** (what every original SIREN run was trained at). The unused legacy
  scalar bias is dropped.

The normalizer is **rebuilt from the training npz** every run (`--train-data`, default `DATA_FILE`,
first `--n-train` cases) — so the `--train-data` you pass must match what the model was trained on.
experiment_name = the checkpoint filename stem (a leading `model_chkpts_` is stripped).

## Outputs

**train** → `CheckPts/<name>.pt` (best-val weights + config) · `Predictions/<name>/loss.txt`
(incremental, includes the `ndiff` column; survives a wall-kill) + `loss.png`.

**test / infer** → under `Predictions/<ckpt stem>/`: per-case metric tables + a `mean ± std`
summary printed to stdout (and `test_log.txt`), plus `traces/`, `snapshots/`+`activation_time/`
(`--snapshot` + `--reference-xyz`), `vtu/<case>/Vm.pvd` (`--vtu-out`), `colorbars/` (`--color-bar`),
`predictions.npz` (`--save`).

## PBS (Vanda; submit from this folder)

| script | run |
|---|---|
| `train.pbs` | plain SIREN, f121 (5 ms), 5000 ep |
| `train_f301.pbs` | plain SIREN, f301 (2 ms), 3000 ep |
| `train_ndiff.pbs` | SIREN + ndiff, f121, scheduled LR (`qsub -v LAM=3` to sweep λ) |
| `train_ndiff_f301.pbs` | SIREN + ndiff, f301, scheduled LR |
