# Geo-DONet (clean)

A geometry-conditioned DeepONet that maps a heart's PCA geometry code **θ → transmembrane
voltage field Vₘ(x, t)** on a fixed reference mesh. This is the consolidated, single-trunk
(Tanh) rewrite — no SIREN, no ndiff loss.

```
main.py    orchestration: parse_args, train(), infer(), main()
opnn.py    architecture: GeoDONet, trunk chunking, checkpoint loading (config from weights)
utils.py   data (load_dataset/load_inputs/split/Normalizer) + evaluation metrics
```

## Setup

```bash
source ~/load_dimon_env.sh        # torch 2.1.2+cu121 venv; sklearn, meshio
```

GPU is auto-detected (`torch.cuda.is_available()` is `False` on login nodes — train on a GPU
node). **Run `main.py` from inside this folder** — all outputs are written relative to the
current working directory (`CheckPts/`, `Predictions/`), so:

```bash
cd Geo_DONet_clean
python main.py ...
```

## The three modes

| mode | command | what it does |
|---|---|---|
| **train** (default) | `python main.py` | train + validate; save best-val weights |
| **test** | `python main.py --test-model --model-path CKPT` | predict the **test split** + evaluate vs GT |
| **infer** | `python main.py --infer --model-path CKPT --data-path NPZ` | predict **all** cases; no GT needed (`--eval` to score) |

```bash
# train with defaults (5000 ep, w300 d4, lr 5e-4)
python main.py

# train, overriding a few knobs
python main.py --epochs 5000 --lr-scheduled --width 300 --batch-size 24

# evaluate on the held-out test split -> per-case tables + mean ± std summary
python main.py --test-model --model-path CheckPts/geodonet_w300_d4_5000ep_lrsched.pt

# evaluate + export figures and the predictions npz
python main.py --test-model --model-path CKPT --snapshot --color-bar --save

# write predicted Vm onto the mesh as a ParaView series, one slow case only
python main.py --test-model --model-path CKPT --vtu-out --mesh ~/scratch/canonical.vtu --cases 100

# predict on a new cohort (no ground truth), save the npz
python main.py --infer --model-path CKPT --data-path new_hearts.npz --save
```

## How configuration works

Every run default lives in the **CONFIGURATION block** at the top of `main.py` (hardcode the
defaults you want), and **every default is exposed as a CLI flag that overrides it** for a
single run. Run with no flags to use the hardcoded config.

Key defaults:

| constant | default | meaning |
|---|---|---|
| `DATA_BASE` | `/home/svu/e1032484/scratch` | one absolute path; npz / xyz / vtu all resolve under it |
| `DATA_FILE` | `geo_donet_data_f121.npz` | default train/infer npz |
| `N_TRAIN, N_VAL` | `95, 5` | split by case order; the rest (25) is the **test set** |
| `WIDTH, DEPTH` | `300, 4` | trunk/branch MLP size |
| `EPOCHS, BATCH_SIZE, LR` | `5000, 24, 5e-4` | optimization |
| `SEED` | `42` | RNG seed |

## Flags

**Training / model**

| flag | default | notes |
|---|---|---|
| `--epochs` | 5000 | |
| `--batch-size` | 24 | |
| `--lr` | 5e-4 | schedule start when `--lr-scheduled` |
| `--lr-scheduled` | off | linear decay `--lr` → 0.1× over all epochs |
| `--width` / `--depth` | 300 / 4 | |
| `--seed` | 42 | |
| `--device` | auto | `cuda` / `cpu` |
| `--model-path` | — | **train**: output ckpt (default `CheckPts/<auto-name>.pt`); **test/infer**: ckpt to load (required) |

**Data + split** (all modes)

| flag | default | notes |
|---|---|---|
| `--data-path` | `DATA_FILE` | absolute path, or a bare name under `DATA_BASE` |
| `--n-train` / `--n-val` | 95 / 5 | split sizes (rest = test) |
| `--train-data` | `DATA_FILE` | npz the normalizer is rebuilt from each run (see below) |

**Inference outputs** (`--test-model` / `--infer`; all opt-in)

| flag | default | notes |
|---|---|---|
| `--cases` | test / all | `all\|train\|val\|test`, or global indices like `100` / `100,101` |
| `--eval` | off | (`--infer` only) score vs GT; `--test-model` always evaluates |
| `--snapshot` | off | Vₘ(t) traces + 3D Vₘ/AT scatter SVGs (3D needs `--reference-xyz`) |
| `--vtu-out` | off | predicted Vₘ `.vtu`+`.pvd` series per case (needs `--mesh`) |
| `--mesh` | `canonical.vtu` | reference `.vtu` (points + tetra) for `--vtu-out` |
| `--reference-xyz` | `canonical_xyz.npz` | cartesian xyz for `--snapshot` 3D scatter |
| `--color-bar` | off | standalone colorbar SVGs |
| `--save` | off | write the full `predictions.npz` |
| `--out-dir` | `Predictions/<ckpt stem>` | override the inference output dir |

## Checkpoints

Checkpoints store **only the model weights** (`{"model_state_dict": ...}`). On every run the
architecture is inferred from the weight shapes and the normalizer is **rebuilt from the
training npz** (`--train-data`, default `DATA_FILE`, fit on the first `--n-train` cases). So:

- There is **one load path** for new and old checkpoints.
- **Legacy** original-`opnn` checkpoints (Geo_DONet, Geo_DONet_ndiff — keys `_branch_g`/`_trunk`
  + an unused scalar bias) load transparently; their keys are remapped automatically.
  *(SIREN checkpoints are not supported — their trunk is a sine layer.)*
- The `--train-data` you pass must match what the model was trained on (default f121, first 95),
  or normalization will be wrong. For `--test-model`, `--data-path` doubles as `--train-data`
  when it already carries GT, so one npz covers both.

experiment_name = the checkpoint filename stem (a leading `model_chkpts_` is stripped); it
drives `Predictions/<experiment_name>/`.

## Outputs

**train** → `CheckPts/<name>.pt` (best-val weights) · `Predictions/<name>/loss.txt` (incremental,
survives a wall-kill) + `loss.png`.

**test / infer** → under `Predictions/<ckpt stem>/`:
- metrics print to stdout — per-case tables **and** a `mean ± std` summary block:
  ```
  results — 25 cases (test):
    V_m Rel L2  0.1482 ± 0.0127      MAE  4.26 ± 0.74 mV
    AT  Rel L2  0.0952 ± 0.0229      MAE  6.15 ± 1.17 ms
  ```
- `traces/` (`--snapshot`) · `snapshots/` + `activation_time/` (`--snapshot` + xyz) ·
  `vtu/<case>/Vm.pvd` (`--vtu-out`) · `colorbars/` (`--color-bar`) · `predictions.npz` (`--save`).

## Data format (npz keys)

- **training** (`--data-path` for `train` / `--test-model`): `theta (n,60)`, `coords (n_nodes,4)`
  Cobiveco, `vm (n,n_nodes,T)`, `time (T,)`, `case_names (n,)` *(optional)*.
- **inference input** (`--infer`): at least `theta` + Cobiveco coords (`coords` or `cobiveco`);
  `vm` / `time` / `case_names` optional (GT only needed with `--eval`).

## Sanity check

Re-evaluating the ndiff λ=1 checkpoint reproduces the recorded numbers exactly, confirming the
legacy load + normalizer rebuild are faithful:

```bash
python main.py --test-model \
  --model-path ../Geo_DONet_ndiff/CheckPts/model_chkpts_geo_donet_5000ep_w300_lrsched_ndiff1.pt
# -> V_m Rel L2 0.1482 ± 0.0127, MAE 4.26 ± 0.74 mV ; AT Rel L2 0.0952 ± 0.0229, MAE 6.15 ± 1.17 ms
```
