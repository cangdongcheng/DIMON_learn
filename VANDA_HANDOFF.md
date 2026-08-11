# DIMON_learn on Vanda — handoff

Temporary working copy while NSCC is down (started 2026-05-29).
Native home is NSCC `BASE/cardiac_simulation/DIMON/` (user `e1590340`).
Active variants: **Geo_DONet** (clean rewrite), **Geo_DONet_SIREN**, **Geo_DONet_ndiff**,
**Geo_DONet_overfit** (single-case capacity test).

> **2026-06-03 — Geo_DONet is now a clean rewrite.** The original messy `Geo_DONet/` was
> deleted and replaced by the from-scratch clean version (formerly `Geo_DONet_clean/`). It is a
> 4-file package: `main.py` (CONFIGURATION + CLI + train/infer), `opnn.py` (architecture +
> checkpoint loaders), `utils.py` (data + evaluation, numpy/torch only), `viz.py` (rendering/IO).
> Modes: train (default) / `--test-model` / `--infer`. Weights-only checkpoints; can still load
> the OLD opnn checkpoints (legacy-key remap). See **Code edits → Geo_DONet/**.

> **2026-06-04 — Geo_DONet_SIREN is now a clean rewrite too** (same 4-file structure as
> Geo_DONet). The 777-line monolithic `main.py` + old `opnn.py` are archived under
> `Geo_DONet_SIREN/_old_monolithic/`; the new package is `main.py`/`opnn.py` + **self-contained
> copies** of Geo_DONet's `utils.py`/`viz.py` (user chose copy over import-coupling). Kept the
> **ndiff loss**, **dropped** the `--trunk xyz` option (Cobiveco-only now, like the baseline),
> dropped the legacy learnable bias. SIREN checkpoints store `{model_state_dict, config}` (config
> carries `omega_0`, which is NOT inferable from weights) — and the loader still reads the OLD
> original-opnn SIREN checkpoints (legacy-key remap, omega_0 assumed 30). `Geo_DONet_ndiff/`
> remains an old-opnn variant. See **Code edits → Geo_DONet_SIREN/**.

> **2026-06-03 — single-case overfit capacity test (`Geo_DONet_overfit/`).** To separate
> *capacity* from *generalization* in the upshoot-smearing problem (prior runs all blamed the
> trunk's spatial representation — see Findings), a new folder overfits the **baseline arch
> (w300 d4 Tanh)** on one heart (case 0). With one geometry the branch collapses to a constant
> vector, so this is a clean test of the trunk coordinate-MLP: can it represent the sharp
> depolarization upstroke *at all*? Run at f121/f301/f601 to separate the Tanh trunk's limit
> from the 5 ms grid's temporal undersampling (the upstroke rises in ~1–2 ms, so at f121's 5 ms
> dt it spans < 1 frame). See **Code edits → Geo_DONet_overfit/**.

> 2026-05-29 (later session): added a ground-truth/prediction **neighbour-difference
> analysis** (`scratch_scripts/`) and a new training variant **Geo_DONet_ndiff** (MSE +
> signed-Laplacian spatial loss). The enabling discovery — `canonical.vtu` is the mesh in
> Vm node order, `reference.vtu` is NOT — is in the **Mesh node order** section below.
> Read it before any edge/adjacency/gradient work.

## Environment

```bash
source ~/load_dimon_env.sh
```

Loader script lives at `~/load_dimon_env.sh`. It loads `Python/3.11.3-GCCcore-12.3.0` (only for `libpython3.11.so`) and activates `~/venvs/dimon`, which has:

| package | version | source |
|---|---|---|
| python | 3.11.3 | EB module |
| torch | 2.1.2+cu121 | pip wheel from download.pytorch.org |
| numpy | 1.26.4 | pip |
| scipy | 1.11.4 | pip |
| scikit-learn | 1.3.2 | pip |
| matplotlib | 3.7.5 | pip |
| meshio | 5.3.5 | pip (reads canonical.vtu; writes .vtu/.pvd outputs) |
| h5py | 3.16.0 | pip (compact XDMF time series; optional) |

**Important:** do NOT use Vanda's EasyBuild module `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`. It was built for `sm_70` only and fails on the A40 (sm_86) with `no kernel image is available for execution on the device`. The pip wheel ships kernels for sm_50–sm_90, which is why the venv uses it instead.

## Data

Currently in `~/scratch/`:

| file | size | shape summary |
|---|---|---|
| `geo_donet_data_f121.npz` | 2.3 GB | 125 hearts × 50797 nodes × 121 frames (5 ms step) |
| `geo_donet_data_f601.npz` | 11 GB  | same, 601 frames (1 ms step) — derive f301 on the fly via `--frame-step 2` |

Keys: `theta (125,60)`, `coords (50797,4)` Cobiveco, `vm (125,50797,T)`, `ecg (125,T,10)`, `time (T,)`, `case_names (125,)`.

Reference mesh / geometry (uploaded 2026-05-29, in `~/scratch/`):

| file | size | what |
|---|---|---|
| `canonical.vtu` | 18 MB | reference tet mesh **in Vm node order** (50797 pts, 254337 tets). Carries `ab/rt/tm/tv/rvlv`, `fiber`, `sheet`, `scar`, `healthy_sim_0..8`, `scar_sim_0..8`. |
| `reference_cobiveco.npz` | 756 KB | `cobiveco (50797,5)` = `canonical.vtu` point_data in order (`labels=[ab,rt,tm,tv,rvlv]`); first 4 cols == `geo_donet` `coords` exactly. |
| `reference.vtu` | 6.6 MB | same mesh, **DIFFERENT node order** — do NOT use for adjacency (see Mesh node order). |
| `reference/mesh_adjacency_data_order.npz` | 2.4 MB | derived: `edge_src/edge_dst` (320286 undirected), `inv_length`, `points_xyz`, `n_nodes`, all in Vm order. Built by `scratch_scripts/build_adjacency_canonical.py`. |

**3D-snapshot cartesian coords without DIMON_training_data_healthy.npz:**
`~/scratch/canonical.vtu` (50797 pts) → `scratch_scripts/extract_canonical_xyz.py` writes `~/scratch/canonical_xyz.npz` (key `cartesian_coords`, 50797×3). The extractor verifies the VTU's ab/rt/tm/tv match the training-npz Cobiveco row-for-row (got exact 0.0 diff → identical node order). `Geo_DONet_SIREN/main.py` has a `load_cartesian_coords()` helper that prefers `DIMON_training_data_healthy.npz` and falls back to `canonical_xyz.npz` (override path via `DIMON_CARTESIAN_NPZ`). This unblocks 3D V_m snapshot + AT-scatter rendering on Vanda. The clean **Geo_DONet/** now uses `canonical_xyz.npz` directly (`--snapshot --reference-xyz`, which defaults to it). `Geo_DONet_ndiff/` still only has the `--skip-snapshots` guard (old opnn).

**Not yet on Vanda** — only needed for the listed cases:

| file | needed by |
|---|---|
| `DIMON_training_data_healthy.npz` | all Cobiveco/* and DIMON_PINN variants (clean Geo_DONet uses `canonical_xyz.npz` instead; it is Cobiveco-only) |
| `DIMON_training_data_healthy_fixed.npz` | DIMON_PINN; Cobiveco/ |
| `Laplace_data{,_supp,_supp2000}.mat` | upstream Laplace example |
| RD / LV example data | upstream ReactionDiffusion, LV |

## Mesh node order — READ BEFORE ANY EDGE/ADJACENCY/GRADIENT WORK

The training data (`geo_donet_data_*.npz` `vm`/`coords`) is in **`canonical.vtu` node
order**, verified exactly (0.0 diff): `canonical.vtu` point_data `ab/rt/tm/tv/rvlv` (f32)
== `reference_cobiveco.npz` == `geo_donet` `coords`. Chain: the cobiveco-extract script
read `canonical.vtu` point_data in order → `reference_cobiveco.npz`; packaging copied that
into `coords` and stacked `vm` in the same order.

`reference.vtu` is the **same mesh in a different node numbering** (in-order max|diff| ≈
0.98). Coordinate-matching it to the data is fuzzy at the apex (rt collapses there) — so
build adjacency from **`canonical.vtu` only**. The adjacency (320286 undirected edges, mean
length 1497.6 µm — matches the ~1500 µm mesh and the old `edge_grad_operator.npz` edge count)
is saved at `~/scratch/reference/mesh_adjacency_data_order.npz`.

**Hypothesis:** `Geo_DONet_grad`'s gradient loss may have "failed" only because its
`edge_grad_operator.npz` was built in a non-Vm order → every `G·vm` differenced
non-adjacent nodes. `Geo_DONet_ndiff` re-tests the idea with the verified adjacency.

### scratch_scripts/ (analysis; run from repo root, CPU is fine)
- `build_adjacency_canonical.py` — builds + verifies the data-order adjacency from `canonical.vtu`.
- `visualize_neighbour_diff.py` — GT Vm neighbour differences for one heart → `.vtu` (static
  summary + `.pvd` series: `ndiff`=mean|ΔVm|, `signed`=Vm−mean(neighbours)) + `gradient_signal.png`
  (Σ_edge (V_d−V_s)² vs t). `--heart` = global index 0..124.
- `compare_pred_neighbour_diff.py` — pred vs GT from a `test_predictions.npz`; overlays the
  gradient-signal curves, writes per-node `ndiff_err`/`signed_err` (candidate loss fields) +
  scalar Vm/ndiff/signed MSEs. `--heart` = GLOBAL index; predictions exist only for test hearts
  100..124. (`build_adjacency_data_order.py`, `inspect_mesh.py`, `check_node_order.py` are earlier
  exploration, superseded.)
- No inference needed for test hearts: `Geo_DONet/Predictions/.../Test/test_predictions.npz`
  already has nodewise `pred`+`true` (mV, canonical order) for the 25 test hearts (global 100..124).

## Code edits

**Geo_DONet/** — clean rewrite (2026-06-03; replaces the original opnn baseline)
- 4 files: `main.py` (CONFIGURATION block + CLI + `train()`/`infer()`), `opnn.py` (`GeoDONet`
  + `chunked_forward` + checkpoint loaders), `utils.py` (data + eval, numpy/torch only),
  `viz.py` (metric tables, traces, 3D scatter SVGs, colorbars, `.vtu`).
- **Modes**: `python main.py` (train), `--test-model` (test split + eval), `--infer` (predict
  any cases; `--eval` to score). All defaults live in the CONFIGURATION block at the top of
  `main.py`; every default has a CLI override (see `--help` / README.md). `DATA_BASE` is a
  hardcoded constant there (no longer an env var).
- **Checkpoints are weights-only** (`{"model_state_dict": ...}`). Architecture is inferred from
  the weight shapes; the normalizer is rebuilt from `--train-data` each run — so ONE load path
  handles both new GeoDONet and **legacy original-opnn** checkpoints (legacy `_branch_g`/`_trunk`
  keys + scalar bias remapped; **SIREN checkpoints NOT supported** — sine trunk).
- **Defaults**: data `geo_donet_data_f601.npz` + `--frame-step 5` → f121; `width 300 depth 4`,
  `5000 ep`, `lr 5e-4` (`--lr-scheduled` → 5e-4→5e-5 linear over epochs), `--patience 500` early
  stop, split 95/5/25. `--grad-checkpoint` bounds GPU memory for many frames (f301/f601).
- **Snapshots work on Vanda**: `--snapshot` uses `--reference-xyz` (default `canonical_xyz.npz`)
  for the 3D scatters; `--vtu-out --mesh canonical.vtu` writes ParaView series; `--n-viz` picks
  which cases to render (global index or `all`; default first 2). Outputs land in
  `Predictions/<ckpt stem>/<type>/<case>/`. No longer needs `DIMON_training_data_healthy.npz`.
- [train.pbs](Geo_DONet/train.pbs), [train_f301.pbs](Geo_DONet/train_f301.pbs) — Vanda PBS headers.

**Geo_DONet_overfit/** — single-case overfit capacity test (2026-06-03)
- **Purpose**: isolate capacity from generalization. Overfit the baseline GeoDONet on one case;
  the branch collapses to a constant vector, so it tests whether the **Tanh trunk** (a
  coordinate-MLP) can represent the sharp V_m upstroke. Outcome logic: fits it → capacity is
  there, the deployed failure is generalization; still smears at **f601** (1 ms, upshoot
  resolved) → spectral bias confirmed → next step is an activation change (SIREN/Fourier
  features), NOT more width/depth (capacity at w300 d4 is already ample for one field).
- [overfit.py](Geo_DONet_overfit/overfit.py) **imports the actual `opnn.py`/`utils.py` from
  `../Geo_DONet/`** (via `sys.path`) so it tests the same arch, never a copy. Full-batch GD on
  the one case; **TF32 on by default** (`--no-tf32` for a full-fp32 run if you need MSE below the
  ~1e-3 TF32 floor); 5000-ep cap + **early stop** (patience 500 on train MSE, min-delta 1e-4
  relative — so a slow spectral-bias asymptote trips the stop). `best.pt` keeps strictly-best
  weights; `last.pt` the final.
- **Five upshoot diagnostics** → `Predictions/case0_f{N}_w300_d4/`: `loss.png` (does train MSE
  → 0?), `frame_error.png` (per-frame Rel L2 — spike at the depolarization frames?),
  **`traces.png`** (GT vs pred V_m(t) for nodes spanning early→late activation — the steepness
  money plot), `at_scatter.png` (AT MAE), `upstroke.png` (max dV/dt GT vs pred), + `overfit_log.txt`.
  Judge on the trace/per-frame plots, NOT the scalar MSE — the upstroke is a tiny volume fraction
  so MSE dilutes it.
- **Three self-contained PBS** (no `-v`/`-l` overrides — submit as-is):
  [overfit_f121.pbs](Geo_DONet_overfit/overfit_f121.pbs) (`--frame-step 5`, 3 h, no checkpoint),
  [overfit_f301.pbs](Geo_DONet_overfit/overfit_f301.pbs) (`--frame-step 2 --grad-checkpoint`, 7 h),
  [overfit_f601.pbs](Geo_DONet_overfit/overfit_f601.pbs) (`--frame-step 1 --grad-checkpoint`, 12 h).
  f301/f601 need grad-checkpoint (too many frame activations for 48 GB); f121 fits without.
  TF32 cut f121 epoch time ~20 % (eta 250→200 min/10k ep on A40).
- **First submission (jobs 1157332/3/4) all died in ~2 s, `Exit_status=1`, 0-byte logs** — NOT a
  Python bug (the script runs fine on the interactive node). Cause: the three PBS files carried
  `set -e`, which aborts on `load_dimon_env.sh`'s first `conda deactivate` (rc=1, silently). Fixed
  2026-06-04 by removing `set -e` from all three (see Vanda PBS conventions). Verified the corrected
  flow reaches Python with all imports loading. **Ready to resubmit.**

**Geo_DONet_SIREN/** — clean rewrite (2026-06-04; mirrors Geo_DONet's 4-file structure)
- 4 files: `main.py` (CONFIGURATION + CLI + `train()`/`infer()`/`main()`), `opnn.py`
  (`GeoDONetSIREN` = Tanh branch + **SineLayer** SIREN trunk, `build_trunk_chunks`/`chunked_forward`,
  checkpoint loaders), and **self-contained copies** of Geo_DONet's `utils.py` (data + eval +
  the **ndiff** `build_laplacian_operator`/`signed_laplacian_loss`) and `viz.py` (unchanged).
  The pre-rewrite monolith (`main.py` 777 lines + old `opnn.py` + `plot_*.py`) is in
  `_old_monolithic/`.
- **Modes** identical to Geo_DONet: train (default) / `--test-model` / `--infer`, same flags
  (`--snapshot`/`--vtu-out`/`--color-bar`/`--save`/`--cases`/`--n-viz`/`--train-data`/`--frame-step`).
- **SIREN-specific over the baseline**: `--omega-0` (default 30; CONFIGURATION `OMEGA_0`),
  `--ndiff-weight λ` + `--adj-file` (default `mesh_adjacency_data_order.npz`), `--grad-checkpoint`
  **on by default** (`GRAD_CHECKPOINT=True` — SIREN trunk activations need it even at f121),
  `CHUNK_FRAMES=10` (vs Tanh's 25), flat LR default **1e-4** (SIREN trains below the Tanh LR).
- **Checkpoints store `{model_state_dict, config}`** — unlike the shape-only GeoDONet checkpoints,
  because `omega_0` is a forward-time multiplier, not a weight, so it can't be inferred from the
  state_dict. Loader: new ckpts use the saved `config`; **legacy original-opnn SIREN ckpts**
  (`_branch_g`/`_trunk.linear`/scalar `bias`, no config) are remapped, arch inferred from shapes,
  `omega_0` defaulted to **30** (what all original SIREN runs used); the unused legacy bias dropped.
- **Naming**: `geodonet_siren_w{w}_d{d}_w0_{ω}_{ep}ep[_lrsched][_ndiff{λ}][_f{frames}]`.
- **PBS** (Vanda, submit from the folder): `train.pbs` (plain f121, 5000 ep), `train_f301.pbs`
  (plain f301, 3000 ep — the stale NSCC header/path is **fixed**), `train_ndiff.pbs` (SIREN+ndiff
  f121, scheduled LR), `train_ndiff_f301.pbs` (SIREN+ndiff f301, 3000 ep). ndiff scripts pass
  `--lr 5e-4 --lr-scheduled` (reproduces the original 5e-4→5e-5) and take `qsub -v LAM=<λ>`.
- **Validated (CPU, 2026-06-04)**: all 4 files compile; `chunked_forward` grad-checkpoint path
  bit-identical to plain; `signed_laplacian_loss` differentiable; **all 3 existing trained SIREN
  checkpoints** (1142478/1147941/1148776) load via the legacy remap; new-format checkpoint
  round-trips with a non-default `omega_0`. NOT yet run on GPU — the first GPU job is also the
  first end-to-end test of the rewrite.

**Geo_DONet_ndiff/** (new variant — copy of Geo_DONet + a spatial loss; 2026-05-29)
- Total loss `= MSE + λ·mean((L·E)²)`, `E=pred−gt`, `L=I−D⁻¹A` (signed neighbour difference on
  the canonical-order mesh). `--ndiff-weight 0` == Geo_DONet baseline exactly.
- [utils.py](Geo_DONet_ndiff/utils.py) — adds `--ndiff-weight λ` and `--adj-file` (default
  `mesh_adjacency_data_order.npz`, env `DIMON_ADJ_FILE`).
- [main.py](Geo_DONet_ndiff/main.py) — `build_laplacian_operator` + `signed_laplacian_loss`; loss
  added to the **train loop only** (val/test stay MSE so best-val selection is comparable to
  baseline); per-epoch log prints `MSE:` and `nDiff:`; `save_directory` gets a `_ndiff{λ}` tag.
- [train.pbs](Geo_DONet_ndiff/train.pbs) — now **λ=1.0** (override `qsub -v LAM=3 train.pbs`),
  w300, 5000 ep. λ=0.1 was tried first and found too low (see Findings). Validated: compiles + CPU
  unit-test of the loss (constant→~0, random→finite & differentiable).

**Not yet ported** (still have NSCC paths in `main.py` and NSCC PBS headers): `Cobiveco/`, `Cobiveco_v2/`, `Cobiveco_with_fiber/`, `Cobiveco_with_scar/`, `DIMON_PINN/`, `Geo_DeepONet/`, `Geo_DONet_grad/`, `Geo_DONet_ECG_joint/`, `ECG_transfer/`, plus upstream `Laplace/`, `ReactionDiffusion/`, `LV/`. Also `main_cv.py` in every ported folder still hardcodes the NSCC `DATA_BASE`.

**Not yet ported:** every other variant directory (`Cobiveco/`, `Cobiveco_v2/`, `Cobiveco_with_fiber/`, `Cobiveco_with_scar/`, `DIMON_PINN/`, `Geo_DeepONet/`, `Geo_DONet_grad/`, `Geo_DONet_ECG_joint/`, `ECG_transfer/`, plus upstream `Laplace/`, `ReactionDiffusion/`, `LV/`) still has hardcoded NSCC paths in `main.py` and NSCC-style PBS headers.

## Vanda PBS conventions (learned this run)

- Header: `#PBS -l select=1:ngpus=1` plus `#PBS -l walltime=...`. No `ncpus`, no `mem`, no `-P`, no `-q`.
- Default GPU is A40 (sm_86, 48 GB) — NSCC's `train.pbs` was tuned for A100, so A100 hyperparams may OOM. If `width=300 --batch-size 24` fails, drop batch size first.
- Submitted jobs end up in the `batch_gpu` queue automatically.
- **Queue walltime limits** (`qstat -Qf`): `batch_gpu` max **168:00:00**, default 24:00:00 — so runs >24h must set `#PBS -l walltime=` explicitly. `interactive_gpu` max 12h. `gpu`/`gpu_amd` max 48h.
- **LR schedule caveat**: `--lr-schedule` is `LinearLR(total_iters=epochs)` — it decays over the *full* `epochs`. A wall-kill before `epochs` completes leaves the model at mid-LR (schedule truncated). Size walltime so the run finishes, or expect a suboptimal best-val. Best-val checkpoint IS saved mid-run, so a kill isn't catastrophic, just suboptimal.
- **NEVER put `set -e` in a PBS script that sources `~/load_dimon_env.sh`** (2026-06-04). The env script is written for *interactive* sourcing: its first action is `conda deactivate 2>/dev/null`, which returns **rc=1** when there's nothing to deactivate (conda IS defined on the compute nodes). Under `set -e` that aborts the job at that line, and the `2>/dev/null` swallows the only message → **exit 1 with a 0-byte `.o` log** (looks like the job vanished). Reproduce: `bash -c 'set -e; source ~/load_dimon_env.sh'` → exit 1, no output. This is exactly what killed the first overfit submissions (jobs 1157332/3/4, all `Exit_status=1` in ~2 s). Every *other* PBS in the repo sources the env without `set -e` and runs fine — match that. Fixed in the three `Geo_DONet_overfit/overfit_f*.pbs`.

## Timing reference (A40, width=300, batch=24)

| run | per-epoch | source |
|---|---|---|
| f121 SIREN + ndiff, 5000 ep | ~0.19 min | `siren_ndiff.o1147941` = 941 min |
| f301 SIREN plain, 3000 ep | ~0.46 min | `Geo_DONet_SIREN.o1142478` = 1378 min |
| f301 SIREN + ndiff (est.) | ~0.5-0.55 min | → 5000 ep ≈ 44-48 h, request 60 h |

## Smoke tests that passed (2026-05-29)

- GPU sanity (matmul + `torch.cuda.get_arch_list()` shows sm_86) on `GN-A40-074`.
- `main.py --width 100 --batch-size 8 --epochs 5 --trunk cobiveco` ran end-to-end on a single A40, ~25 s ETA per epoch, loss curve written to `Predictions/geo_donet_5ep_w100/loss_curve.png`.
- Geo_DONet_ndiff `main.py`/`utils.py` compile; CPU unit-test of `signed_laplacian_loss` passes (constant field → ~1e-15, random → finite & differentiable). GPU run still pending.

## Findings — ndiff loss (2026-05-29)

First `Geo_DONet_ndiff` run at **λ=0.1** (`gd_ndiff.o1147597`): the `nDiff` term sat **flat at
its random-init value ~0.0147 through epoch 2760** while train MSE fell ~37× (1.68 → 0.043).
Two reads, both acted on:
1. **λ too low** — `λ·nDiff ≈ 0.0015` was only ~3 % of MSE, so no real gradient pressure.
   → bumped `Geo_DONet_ndiff` to **λ=1.0** (`λ·nDiff ≈ MSE` would need λ≈3; sweep 1/3/10 via `-v LAM=`).
2. **Spectral bias (likely the real cause)** — nDiff pinned at the GT's *own* Laplacian energy
   means the smooth Tanh trunk emits ~zero spatial Laplacian: it fits the smooth part of Vm but
   never the sharp wavefront, so the Laplacian-of-error can't fall regardless of λ. → the ndiff
   loss was ported to the **SIREN trunk** (`Geo_DONet_SIREN/train_ndiff.pbs`, f121) to test whether
   a high-frequency-capable net drives nDiff down where Tanh could not. Compare the two `nDiff`
   curves (Tanh-ndiff vs SIREN-ndiff, both f121).

## Results — test-set eval (25 hearts, mean ± std)

| Run | trunk | data | λ | ep | V_m Rel L2 | V_m MAE (mV) | AT Rel L2 | AT MAE (ms) |
|---|---|---|---|---|---|---|---|---|
| baseline | Tanh | f121 | 0 | 5000 | (val MSE 0.0399) | — | — | — |
| siren+ndiff f121 (`1147941`) | SIREN | f121 | 1 | 5000 | 0.143 | 3.99 | — | — |
| siren+ndiff **f301** (`1148776`) | SIREN | f301 | 1 | 3000 | **0.1442 ± 0.0204** | **3.97 ± 0.88** | **0.0822 ± 0.0225** | **5.13 ± 1.10** |
| ndiff λ1 (`1149334`) | Tanh | f121 | 1 | 5000 | 0.1482 ± 0.0127 | 4.26 ± 0.74 | 0.0952 ± 0.0229 | 6.15 ± 1.17 |
| ndiff λ3 (`1149335`) | Tanh | f121 | 3 | 5000 | 0.1498 ± 0.0136 | 4.31 ± 0.98 | 0.0960 ± 0.0255 | 6.17 ± 1.39 |

**Finding (2026-06-01): Tanh + ndiff did NOT help, confirming spectral bias.** Evaluated the
two Tanh λ=1/λ=3 checkpoints (f121, 5000 ep). V_m Rel L2 **0.1482** (λ1) / **0.1498** (λ3) —
both *worse* than the SIREN+ndiff f121 run (0.143) and the higher λ marginally worse than the
lower, i.e. adding spatial-Laplacian pressure on a Tanh trunk does nothing or slightly hurts.
This matches the training logs where `nDiff` stayed pinned at its random-init/GT value ~0.0142
at λ=0.1, 1, AND 3 — the smooth Tanh trunk emits ~zero spatial Laplacian, so the loss is inert
regardless of weight. AT MAE ~6.15–6.17 ms (worse than SIREN-f301's 5.13 ms). **Net: neither a
high-freq trunk (SIREN), finer temporal sampling (f301), nor a spatial-Laplacian loss (ndiff)
meaningfully sharpened the wavefront — the bottleneck is the trunk's spatial representation.**

**Finding (2026-05-31): f301 did NOT help.** Going from f121 (5 ms) to f301 (2 ms) temporal
resolution left V_m Rel L2 essentially unchanged (0.1442 vs 0.143 — marginally *worse*, within
noise). The extra temporal samples over the upstroke did not sharpen the depolarizing wavefront
at the metric level. Consistent with the spectral-bias read below: the bottleneck is the trunk's
spatial representation of the sharp wavefront, not temporal sampling. AT metrics (~8 % Rel L2,
~5 ms MAE) are the first AT numbers recorded on Vanda but have no f121 counterpart yet to compare.

## Live jobs
- `1142468` — Geo_DONet `--trunk cobiveco` 5000 ep — **DONE** (633 min). Best val MSE **0.03994**; train→0.025 but val drifted to ~0.059 (overfits past best). Checkpoint `Geo_DONet/CheckPts/model_chkpts_geo_donet_5000ep_w300_lrsched.pt`; eval already wrote `Predictions/geo_donet_5000ep_w300_lrsched/Test/test_predictions.npz` (pred+true, 25 test hearts).
- `1142478` — Geo_DONet_SIREN f601→f301 ω₀=30, 3000 ep — **DONE** (1378 min, ~0.46 min/ep). Checkpoint `..._3000ep_w300_w0_30_f301.pt`.
- `1147597` — Geo_DONet_ndiff **λ=0.1** — ran; nDiff stayed flat at ~0.0147 (see Findings), effectively the baseline (ndiff inactive). Superseded by the λ=1.0 config.
- `1147941` — Geo_DONet_SIREN **+ndiff** (λ=1, lr-sched, f121) 5000 ep — **DONE** (941 min). Test Rel L2 **0.143** / MAE **3.99 mV**. Checkpoint `..._5000ep_w300_w0_30_lrsched_ndiff1_f121.pt`. f121 (5 ms) V_m traces still smear the sharp depolarizing wavefront → motivates the f301 run below.
- `1148776` — Geo_DONet_SIREN **+ndiff λ=1, f301** 3000 ep — **DONE**. Eval'd 2026-05-31: V_m Rel L2 **0.1442 ± 0.0204**, MAE 3.97 mV; AT Rel L2 0.0822, MAE 5.13 ms. **No improvement over f301-plain or f121-ndiff** — see Results table. Checkpoint `..._3000ep_w300_w0_30_lrsched_ndiff1_f301.pt`. Note: earlier mis-launches `1148787`/`1148788` (SIREN, ndiff OFF, wrong-dir qsub) were killed.
- **clean Geo_DONet** (2026-06-03, jobs `1152842` train / `1152853` f301) — the rewrite is
  trained + eval'd. `geodonet_w300_d4_5000ep_lrsched.pt` (Tanh, f121) `--test-model` (25 test):
  V_m Rel L2 **0.1518 ± 0.0251**, MAE 4.73 mV; AT Rel L2 0.0932, MAE 6.03 ms — i.e. it reproduces
  the old baseline architecture (≈ the ndiff numbers), confirming the clean port is faithful.
  Caveat: an f301-trained checkpoint eval'd on the **f121** default grid reads ~0.1624, but that
  is a normalization mismatch (vm_scale differs by frame sampling) — for a clean f301 eval pass
  `--data-path geo_donet_data_f601.npz --frame-step 2` so data + normalizer are both f301.
- **Geo_DONet_overfit** (2026-06-03) — single-case overfit capacity test, baseline w300 d4 Tanh,
  case 0, at f121/f301/f601. Harness built + validated (compiles, smoke ran on the A40 interactive
  node); f121 ~200 min/5000 ep with TF32. First `qsub` (jobs `1157332`/`1157333`/`1157334`,
  2026-06-03) **all failed in ~2 s with `Exit_status=1` and 0-byte logs** — the `set -e` +
  `conda deactivate` interaction, NOT the experiment (fixed 2026-06-04, see PBS conventions).
  **Still PENDING results** — resubmit `qsub overfit_f{121,301,601}.pbs`. f601 is the decisive
  read: if `traces.png` still smears the upstroke there, the Tanh trunk is the limit
  (→ SIREN/Fourier next), not capacity or sampling.
- `1149334` / `1149335` — Geo_DONet_ndiff **λ=1 / λ=3** (Tanh, f121) 5000 ep — **DONE** (651 min each). `nDiff` stayed pinned at ~0.0142 at both λ (never fell vs λ=0.1). Best val MSE 0.039871 (λ1) / 0.040139 (λ3) ≈ baseline 0.03994. Eval'd 2026-06-01 (`--test-model 1 --skip-snapshots`, job 1150915): V_m Rel L2 **0.1482 ± 0.0127** (λ1) / **0.1498 ± 0.0136** (λ3); AT MAE 6.15 / 6.17 ms — see Results table + 2026-06-01 finding. Checkpoints `..._ndiff1.pt`, `..._ndiff3.pt`; `test_predictions.npz` in `Predictions/..._ndiff{1,3}/Test/`.

## Open items

1. Run the ndiff experiments (smoke-test 5 ep, then `qsub`): **Geo_DONet_ndiff λ=1.0** (`train.pbs`) and **Geo_DONet_SIREN +ndiff λ=1.0** (`train_ndiff.pbs`, f121). Key comparison: does `nDiff` fall on SIREN where it stayed flat on Tanh? Success = best-val MSE beats 0.03994 and/or the neighbour-diff error shrinks (re-run `compare_pred_neighbour_diff.py` on each `--test-model 1` output). Sweep λ ∈ {1,3,10} via `qsub -v LAM=`.
2. `DIMON_training_data_healthy*.npz` still not on Vanda — needed only for the Cobiveco*/DIMON_PINN variants now. The clean Geo_DONet is Cobiveco-only and uses `canonical_xyz.npz` for snapshots, so it no longer needs it.
3. Port `main_cv.py`'s `DATA_BASE` in the ported folders if you want cross-validation runs.
4. **Run the single-case overfit capacity test** (`Geo_DONet_overfit/`): `qsub overfit_f{121,301,601}.pbs`. Decision rule: if f601 (upshoot temporally resolved) still smears the upstroke in `traces.png`/`frame_error.png`, the Tanh trunk lacks the representation → switch activation (SIREN/Fourier features), not width/depth. If it fits the upshoot, the deployed failure is generalization, not capacity — pivot to data/conditioning/loss. This directly tests the spectral-bias hypothesis the Findings section has been inferring indirectly.
5. Decide whether to keep this fork on Vanda or rebase onto NSCC once it's back — edits stay narrow (DATA_BASE env vars, CLI args, PBS rewrites, the new Geo_DONet_ndiff + Geo_DONet_overfit variants + scratch_scripts/), so reverse-porting is cheap. Note the new pip deps (meshio, h5py).
