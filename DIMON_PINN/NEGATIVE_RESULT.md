# PINN-on-DIMON: documented negative result

**Date investigated:** 2026-04-11 → 2026-04-12
**Status:** PINN approach abandoned for current data preprocessing. Data-only DIMON variants are the recommended baseline.

This document records why adding an Eikonal PDE residual loss to a DIMON-style neural operator does not work on our current training data, and what the underlying mechanism is. It supersedes the earlier hypothesis (mesh-too-coarse) and explains why the published DIMON recipe deliberately omits a physics loss term.

---

## 1. Hypothesis being tested

DIMON learns activation-time fields on a canonical reference mesh by pulling each patient's openCARP solution onto a single shared template via Cobiveco coordinates. The earlier PINN experiment ([DIMON_PINN/main.py](main.py)) added an anisotropic Eikonal residual term:

```
R(x) = sqrt( ∇t(x)^T · M(x) · ∇t(x) ) - 1
loss = data_MSE + λ · |R|²
```

with `M = vl² ff^T + vt² ss^T + vn² nn^T` from the per-node fiber basis. Multiple `λ` schedules were tried, all of which either crashed training or hurt prediction accuracy.

The original hypothesis (recorded in [CLAUDE.md](CLAUDE.md)) was that the **1500 µm training mesh was too coarse for the PDE gradients to be meaningful**. The plan: solve the eikonal at finer resolutions (1000 / 750 / 600 µm), recompute the residual, and check whether it shrinks. If yes, regenerate training data at a finer resolution and retry the PINN.

## 2. The investigation

Single-case Eikonal solutions on **IMMC001** at four mesh resolutions, all with `vl/vt/vn = 600/240/240 µm/ms` (matching [SM_eikonal.py](../../SM_eikonal.py) and the openCARP convention):

| Mesh | Nodes | Solve time | `\|R\|` median (k-NN, native mesh) |
|---|---|---|---|
| 1500 µm | 50,797 | 40 s | **0.0935** |
| 1000 µm | 97,164 | 2 min | **0.0885** |
| 750 µm | 307,693 | 10 min | **0.0879** |
| 600 µm | 643,341 | 14 s gradient + ~8 min total | **0.0883** |
| 400 µm | 1,981,976 | did not converge in 2h+ | — |

Methodology — see [physics_residual.py](../../physics_residual.py):
- Per-node `∇t` via 15-NN least-squares against neighbour offsets
- Per-node velocity tensor from [package_eikonal.py](../../package_eikonal.py) (element-wise tensor averaged onto incident nodes)
- Mask `t > 5 ms` to drop stim-region nodes
- Median (not mean) across nodes — robust to the long outlier tail caused by `-1` unactivated nodes near the periphery and tiny-edge artifacts
- Validated independently with an MLP-proxy + autograd method (same residual to within fitting noise on 1500 µm; underparameterized at finer resolutions, so not load-bearing for the cross-resolution comparison)

**The discrete eikonal residual is flat across a 13× node-count refinement.** Refining the mesh did **not** reduce the residual that the PDE-loss term is asking the network to satisfy. This refutes the original hypothesis.

## 3. The smoking gun: native vs projected residual

Re-checking the residual on the original DIMON training data ([test_dimon_residual.py](../../test_dimon_residual.py)) — i.e. on the canonical reference mesh, with the activation field projected from per-patient meshes via Cobiveco:

| Source | Velocity | `\|R\|` mean | `\|R\|` median |
|---|---|---|---|
| Native IMMC001 1500 µm | vl=600 | 0.120 | **0.094** |
| **Canonical reference** (heart 0, pacing 0) | vl=640 (old buggy) | 0.376 | 0.298 |
| **Canonical reference** (heart 0, pacing 0) | vl=600 (corrected) | 0.343 | 0.279 |
| Canonical reference (heart 0, pacing 1) | vl=600 | 0.314 | 0.262 |
| Canonical reference (heart 0, pacing 2) | vl=600 | 0.317 | 0.243 |
| Canonical reference (heart 10, pacing 0) | vl=600 | 0.337 | 0.268 |
| Canonical reference (heart 50, pacing 0) | vl=600 | 0.392 | 0.304 |

**Roughly 3× the residual on the projected canonical mesh as on the native patient mesh**, holding velocity, pacing site, and patient identity constant. Velocity calibration shaves ~10% relative; the rest (~70% of the original 0.376) is the projection.

## 4. Mechanism — three identified problems

### Problem 1 (dominant): wrong fiber field in the PINN loss

All 125 patients' fiber fields were derived independently on their own meshes via Cobiveco-based rule-based assignment (Doste 2019). Each patient's eikonal solution was computed using that patient's own anisotropic velocity tensor `M_patient = vl² f_patient f_patient^T + ...`.

But the PINN loss in `DIMON_PINN/main.py` computes the residual using `ref_anisotropy` — a **single** fiber field from the canonical reference mesh — for all 125 patients. Patient 50's wavefront followed patient 50's fibers, but the residual check uses the reference's fibers. The velocity tensor is simply wrong for every patient except whichever one happens to be closest to the reference anatomy.

This alone is sufficient to explain the ~30% residual. Different fiber orientations → different velocity tensors → the wavefront travels in directions that don't match where `M_ref` says it should → large `|sqrt(∇t^T M_ref ∇t) - 1|`.

### Problem 2 (secondary): gradient not transferred through the diffeomorphism

DIMON's diffeomorphic mapping via Cobiveco pulls each patient's activation field onto the canonical reference `Ω₀`: `t̃(x̃) = t(φ_θ⁻¹(x̃))`. For scalar values this is correct — the activation time is preserved point-by-point.

But the eikonal residual involves the **gradient** `∇t`, which transforms under a diffeomorphism:

| Quantity | Transformation under `φ` |
|---|---|
| `t` | `t̃(x̃) = t(φ⁻¹(x̃))` — preserved (scalar) |
| `∇t` | `∇̃t̃ = J_φ⁻ᵀ ∇t` — picks up a Jacobian factor |
| `M` (covariant 2-tensor) | `M̃ = J_φ M J_φᵀ` — picks up two Jacobian factors |

The PINN loss computes `∇t̃` with respect to the reference mesh's Cartesian coordinates — this is **not** the Jacobian-corrected gradient `J_φ⁻ᵀ ∇t`. The diffeomorphic mapping assumes local linearity, but the gradient operator on the reference does not account for the deformation.

Even if problem 1 were fixed (per-patient fiber fields), problem 2 would still contribute some residual from the uncorrected gradient. However, this is expected to be a smaller effect than the fiber mismatch — the Cobiveco-based correspondence is relatively smooth, so the Jacobian is close to orthogonal and the gradient distortion is a few percent, not ~30%.

### Problem 3 (fundamental): data-driven is sufficient for this problem

Physics-informed losses are most useful when **data is scarce** and the PDE must fill in information the data doesn't provide — e.g., a handful of sensor readings on a domain where you need the full field, or a single geometry where you want to generalize to unseen boundary conditions.

Our setting is the opposite: **data is abundant**. 125 hearts × 9 pacing sites = 1,125 full-field solutions, each with 50,797 nodes = ~57 million supervised data points. The data-MSE loss already tells the network everything it needs to know about how wavefronts propagate through anisotropic tissue. The eikonal PDE adds no information the data doesn't already contain.

The DIMON paper's own results confirm this: Yin et al. trained on 1,006 hearts × 7 pacings ≈ 7,000 samples with **pure data-MSE** and achieved ~20 ms max error and ~1.8% relative L² error on activation times. No physics term was needed, and none was used.

Adding a PDE loss in this data-rich regime does not improve generalization — it only introduces a conflicting training signal (problems 1 and 2 above), because the representation on the canonical reference is designed for value prediction, not for satisfying differential equations.

**Summary of the mechanism:**

| Ingredient | Status in PINN loss | Effect |
|---|---|---|
| `t̃` (activation) | Correctly transferred via Cobiveco | OK |
| `M` (velocity tensor) | **Wrong** — uses reference fibers for all 125 patients | **~30% residual (dominant)** |
| `∇t̃` (gradient) | **Uncorrected** — reference-Cartesian, no Jacobian | Additional few-% residual |
| Physics constraint itself | **Unnecessary** — data is abundant enough to learn the mapping without PDE enforcement | Adds conflicting signal, no benefit |

## 5. Why the published DIMON paper doesn't have this problem

The paper ([Yin et al. 2024 → reading notes](../../literature/dimon_notes.md)) trains DIMON with **only the data-MSE loss** — no PDE residual term. This is the correct design choice for two reinforcing reasons:

1. **The canonical-reference representation is value-preserving but not structure-preserving.** The framework's universal approximation result (Theorem 1) only requires continuity of the solution operator under the diffeomorphism — a much weaker condition than PDE residual preservation. The authors understood this.

2. **The data regime doesn't need physics.** With ~7,000 full-field supervised samples (1,006 hearts × 7 pacings), the network has enough data to learn the geometry → activation mapping directly. Physics constraints add value when data is scarce, not when it's abundant. The authors had sufficient data and never needed the PDE as a regularizer.

We attempted to add a physics loss and discovered why the authors didn't — both the representation and the data regime work against it.

## 6. Implications and recommendations

### Don't pursue PINN on the current preprocessing
The data-MSE-only DIMON variants ([Cobiveco/](../Cobiveco), [Cobiveco_v2/](../Cobiveco_v2), [Cobiveco_with_fiber/](../Cobiveco_with_fiber)) are operating exactly the way the framework is designed. They are the recommended baseline going forward. The PDE-residual term should be removed from [DIMON_PINN/main.py](main.py), or the folder retired entirely.

### If you ever want a physics loss, two viable paths
Both are substantial reworks — flagged here so future-you doesn't waste time on the obvious-but-wrong fix (regenerate training data at a finer resolution).

1. **Jacobian-corrected residual on the canonical reference.** Store the per-patient `J_θ` (or reconstruct it from Cobiveco) alongside `theta`. At each query point, transform `∇t̃` and `M_ref` back through `J_θ` and compute the residual in the patient frame. Tractable in principle but requires storing or recomputing the Jacobian field, and the canonical-mesh fibers become irrelevant — at which point the framework loses much of its advantage.
2. **Train on per-patient native meshes.** Each patient is its own instance with its own `(xyz, t, M)`. Native residual is ~9% (the irreducible discretization floor), so the PINN loss is fighting a much smaller signal. Price: you lose DIMON's fixed-domain MIONet structure and need a variable-mesh architecture (PointNet-style branches, mesh-agnostic operators, etc.). This is a significant architectural pivot, not a tweak.

### Update the project memory
The memory entry [project_dimon_pinn](../../../.claude/projects/-home-users-nus-e1590340/memory/project_dimon_pinn.md) currently says the PINN failed because the 1500 µm training data was too coarse. That hypothesis is **wrong** — the residual is flat at ~9% across all native mesh resolutions tested. The real story is the projection-onto-canonical, not the mesh resolution. Update accordingly.

## 7. Reproducibility

To re-derive the numbers in this document:

```bash
# Native-mesh residuals (run package + residual on each resolution)
source ~/load_opencarp_env.sh
python SM_eikonal.py --case IMMC001_20150716_merge_000 --variant healthy --ID 1500_resolution_eikonal
# repeat for 1000, 750, 600 (edit the mesh basename in SM_eikonal.py per resolution)
python package_eikonal.py --case IMMC001_20150716_merge_000 \
    --mesh IMMC001_20150716_merge_000_healthy --sim 1500_resolution_eikonal_healthy

# Residual computation (GPU recommended for the proxy method)
source ~/load_dimon_env.sh
python physics_residual.py --npz <path>/1500_resolution_eikonal_healthy/1500.npz

# Canonical-reference residual (the projection check)
python test_dimon_residual.py
```

All scripts referenced live at the BASE root. The 1500 → 600 µm Eikonal solves take a combined ~25 minutes on a single CPU core.

## 8. One side observation worth keeping

The MLP proxy method ([physics_residual.py](../../physics_residual.py) `proxy_residual`) gets **worse** as the mesh gets finer, not because the underlying residual changes but because the fixed 3 → 256 → 256 → 1 architecture is underparameterized for a 643k-node field. Final training MSE on the 600 µm run was ~5.3 ms² after 20k epochs — still actively underfitting. This is a useful sanity check on the proxy method's operating envelope: it works fine for ≤100k-node meshes but cannot be trusted as an independent residual estimator on the larger meshes. The discrete k-NN method is the load-bearing one for the cross-resolution comparison.
