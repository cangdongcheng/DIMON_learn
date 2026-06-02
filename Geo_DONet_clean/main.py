"""
Geo-DONet (clean) — train / test / infer entry point.

Three functional modules:
  - architecture : opnn.GeoDONet / build_trunk_chunks / chunked_forward
  - data         : utils.load_dataset / load_inputs / split_indices / Normalizer
  - evaluation   : utils.mse (train) + vm_rel_l2_mae / activation_time (inference)

Every run default lives in the CONFIGURATION block below; each is exposed as a CLI
flag that overrides it. Run with no flags for the hardcoded config; pass a flag to
override it. Run main.py from the folder it lives in — outputs are written relative
to the cwd (CheckPts/, Predictions/<experiment_name>/).

main() dispatches three modes (default = train):
  - (no flag)     train()                 train + validate; save the best-val weights
  - --test-model  infer(test_mode=True)   forward pass on the test split + evaluate vs GT
  - --infer       infer(test_mode=False)  forward pass on all cases; --eval to also score

Checkpoints are just model weights. The architecture is inferred from the weight
shapes and the normalizer is rebuilt from the training npz (--train-data) on every
run, so there is one load path for both new (GeoDONet) and legacy (original-opnn)
checkpoints — legacy keys are remapped transparently. experiment_name = the
checkpoint filename stem, and all outputs land in Predictions/<experiment_name>/.

Usage:
    python main.py                                              # train (defaults)
    python main.py --epochs 5000 --lr-scheduled --width 300     # train, overrides
    python main.py --test-model --model-path CheckPts/<run>.pt  # eval on the test split
    python main.py --test-model --model-path <ckpt> --snapshot --save --color-bar
    python main.py --test-model --model-path <ckpt> --vtu-out --mesh canonical.vtu --cases 100
    python main.py --infer --model-path <ckpt> --data-path new.npz --save   # predict, no GT
    python main.py --infer --model-path <ckpt> --data-path new.npz --eval   # predict + score
"""
import os
import argparse
import time as timer
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
from torch.optim.lr_scheduler import LinearLR

from utils import (load_dataset, load_inputs, split_indices, Normalizer,
                   mse, vm_rel_l2_mae, activation_time, at_rel_l2_mae, to_numpy)
from opnn import (GeoDONet, build_trunk_chunks, chunked_forward, build_model,
                  is_legacy_state_dict, remap_legacy_state_dict, config_from_state_dict)


# ════════════════════════ CONFIGURATION ════════════════════════
# Defaults for every run. Each is exposed as a CLI flag (below) that overrides it.

# where the data lives — one absolute path; the npz, reference xyz (--snapshot) and
# reference vtu (--vtu-out) all resolve under it.
DATA_BASE = "/home/svu/e1032484/scratch"
DATA_FILE = "geo_donet_data_f121.npz"                  # default train/infer npz
REFERENCE_XYZ = os.path.join(DATA_BASE, "canonical_xyz.npz")  # xyz for --snapshot 3D scatter
REFERENCE_VTU = os.path.join(DATA_BASE, "canonical.vtu")      # mesh (cells) for --vtu-out

# train / val / test split (by case index order; rest after train+val = test)
N_TRAIN, N_VAL = 95, 5

# model architecture
WIDTH, DEPTH = 300, 4

# optimization
EPOCHS, BATCH_SIZE, LR, LR_SCHEDULED = 5000, 24, 5e-4, False
SEED = 42
DEVICE = None                                          # None -> auto (cuda if available)

# inference outputs — all opt-in. Evaluation is on for --test-model, or --eval for --infer.
SNAPSHOT = False                                       # V_m(t) traces + 3D V_m/AT scatter SVGs
VTU_OUT = False                                        # predicted V_m .vtu+.pvd series (needs --mesh)
COLOR_BAR = False                                      # standalone colorbars
SAVE = False                                           # write predictions.npz

# fixed internals (rarely changed; no CLI flag)
CHUNK_FRAMES = 25                                      # trunk frames evaluated per forward chunk
AT_THRESHOLD_MV = -10.0                                # activation = first upward crossing
VM_MIN, VM_MAX = -90.0, 50.0                           # V_m colour scale (mV)
VM_ERR_MIN, VM_ERR_MAX = 0.0, 20.0                     # |dV_m| colour scale (mV)
CMAP_VM = CMAP_AT = "RdYlBu_r"
CMAP_ERR = "Reds"


def parse_args():
    p = argparse.ArgumentParser(description="Geo-DONet (clean) — train / test / infer")

    # ── mode (mutually exclusive; default = train) ──
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--test-model", action="store_true",
                      help="forward pass on the test split + evaluate against GT")
    mode.add_argument("--infer", action="store_true",
                      help="forward pass on all cases of --data-path (add --eval to score)")

    # ── training + model (defaults from the CONFIGURATION block) ──
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR, help="learning rate (schedule start)")
    p.add_argument("--lr-scheduled", action=argparse.BooleanOptionalAction, default=LR_SCHEDULED,
                   help="linearly decay LR from --lr to 0.1x over all epochs")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", type=str, default=DEVICE, help="cuda/cpu (default: auto)")
    p.add_argument("--model-path", type=str, default=None, metavar="PATH",
                   help="checkpoint path. train: output (default CheckPts/<auto-name>.pt); "
                        "test/infer: checkpoint to load (required)")

    # ── data + split (shared by all modes) ──
    p.add_argument("--data-path", dest="data_file", type=str, default=None, metavar="PATH",
                   help=f"npz: absolute path or a bare name under DATA_BASE (default {DATA_FILE})")
    p.add_argument("--n-train", type=int, default=N_TRAIN, help="train split size")
    p.add_argument("--n-val", type=int, default=N_VAL, help="val split size (rest = test)")
    p.add_argument("--train-data", type=str, default=None, metavar="PATH",
                   help=f"npz the normalizer is rebuilt from each run (first --n-train cases; "
                        f"default {DATA_FILE}, or --data-path when it already carries GT)")

    # ── inference outputs (all opt-in) ──
    p.add_argument("--cases", type=str, default=None, metavar="SEL",
                   help="cases to run inference on: all|train|val|test, or global indices "
                        "like 100 / 100,101 (default: test for --test-model, all for --infer)")
    p.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False,
                   help="(--infer only) compute metrics vs GT; errors if GT absent. "
                        "--test-model always evaluates.")
    p.add_argument("--snapshot", action=argparse.BooleanOptionalAction, default=SNAPSHOT,
                   help="V_m(t) traces + 3D V_m/AT scatter SVGs (3D needs --reference-xyz)")
    p.add_argument("--vtu-out", action=argparse.BooleanOptionalAction, default=VTU_OUT,
                   help="predicted V_m .vtu+.pvd series per case (needs --mesh)")
    p.add_argument("--mesh", type=str, default=REFERENCE_VTU, metavar="PATH",
                   help="reference .vtu (points + tetra cells) for --vtu-out")
    p.add_argument("--reference-xyz", type=str, default=REFERENCE_XYZ, metavar="PATH",
                   help="cartesian xyz npz (key 'cartesian_coords') for --snapshot 3D scatter")
    p.add_argument("--color-bar", dest="color_bar", action=argparse.BooleanOptionalAction,
                   default=COLOR_BAR, help="save standalone colorbars")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=SAVE,
                   help="write the full predictions.npz")
    p.add_argument("--out-dir", type=str, default=None,
                   help="override the output dir (default Predictions/<ckpt stem>)")
    return p.parse_args()


def resolve_data_path(data_file):
    """None -> DATA_BASE/DATA_FILE; absolute -> as-is; bare name -> under DATA_BASE."""
    if data_file is None:
        return os.path.join(DATA_BASE, DATA_FILE)
    return data_file if os.path.isabs(data_file) else os.path.join(DATA_BASE, data_file)


def checkpoint_stem(model_path):
    """experiment_name for a checkpoint: drop the directory, the .pt suffix, and a
    leading 'model_chkpts_' (legacy naming). Drives Predictions/<stem>/.
    e.g. CheckPts/model_chkpts_geo_donet_5000ep_w300_lrsched.pt -> geo_donet_5000ep_w300_lrsched"""
    base = os.path.splitext(os.path.basename(model_path))[0]
    return base[len("model_chkpts_"):] if base.startswith("model_chkpts_") else base


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════ training ════════════════════════════
def train(args, device):
    def to_device(array):
        return torch.tensor(array, dtype=torch.float32, device=device)

    # ── data ──────────────────────────────────────────────────────────────
    data_path = resolve_data_path(args.data_file)
    if not os.path.exists(data_path):
        raise SystemExit(f"data not found: {data_path}")
    data = load_dataset(data_path)
    theta, coords, vm, time = data["theta"], data["coords"], data["vm"], data["time"]
    n_cases, n_nodes, n_frames = vm.shape
    train_idx, val_idx, test_idx = split_indices(n_cases, args.n_train, args.n_val)
    print(f"data: {data_path}\n  {n_cases} cases, {n_nodes} nodes, {n_frames} frames | "
          f"{len(train_idx)} train / {len(val_idx)} val ({len(test_idx)} test held out)")

    # fit normalization on the train split only, then load just the splits we train
    # on. Test is never normalized or moved to the device — it is an inference input.
    normalizer = Normalizer(theta[train_idx], vm[train_idx], coords, time)
    theta_train = to_device(normalizer.theta(theta[train_idx]))
    vm_train = to_device(normalizer.vm(vm[train_idx]))
    theta_val = to_device(normalizer.theta(theta[val_idx]))
    vm_val = to_device(normalizer.vm(vm[val_idx]))
    coords_norm = to_device(normalizer.coords(coords))      # (n_nodes, coord_dim)
    time_norm = to_device(normalizer.time(time))            # (n_frames,)
    trunk_chunks = build_trunk_chunks(coords_norm, time_norm, CHUNK_FRAMES)

    # ── model ─────────────────────────────────────────────────────────────
    model = GeoDONet(geo_dim=theta.shape[1], coord_dim=coords.shape[1],
                     width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, 1.0, 0.1, args.epochs) if args.lr_scheduled else None
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"model: GeoDONet w{args.width} d{args.depth} | {n_parameters:,} params | {device}")

    # output paths: checkpoint (best-val weights) -> CheckPts/<name>.pt;
    # loss log + curve -> Predictions/<name>/.  experiment_name = checkpoint stem.
    default_name = f"geodonet_w{args.width}_d{args.depth}_{args.epochs}ep"
    default_name += "_lrsched" if args.lr_scheduled else ""
    default_name += "" if args.seed == SEED else f"_seed{args.seed}"
    checkpoint_path = args.model_path or os.path.join("CheckPts", f"{default_name}.pt")
    experiment_name = checkpoint_stem(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    pred_dir = os.path.join("Predictions", experiment_name)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"checkpoint -> {checkpoint_path} | logs -> {pred_dir}/")

    # ── train (loss log written incrementally so a wall-killed run keeps its log) ──
    history = {"train": [], "val": []}
    best_val_mse = float("inf")
    train_order = np.arange(len(train_idx))
    start_time = timer.time()
    with open(os.path.join(pred_dir, "loss.txt"), "w") as loss_log:
        loss_log.write("# epoch\ttrain_mse\tval_mse\n")
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(train_order)
            running_loss, n_batches = 0.0, 0
            for batch_start in range(0, len(train_order), args.batch_size):
                batch = train_order[batch_start:batch_start + args.batch_size]
                loss = mse(chunked_forward(model, theta_train[batch], trunk_chunks, n_nodes),
                           vm_train[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            if scheduler:
                scheduler.step()
            train_mse = running_loss / n_batches

            model.eval()
            with torch.no_grad():
                val_mse = mse(chunked_forward(model, theta_val, trunk_chunks, n_nodes),
                              vm_val).item()
            history["train"].append(train_mse)
            history["val"].append(val_mse)
            loss_log.write(f"{epoch}\t{train_mse:.8f}\t{val_mse:.8f}\n")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            if epoch % 10 == 0:
                loss_log.flush()
                elapsed = timer.time() - start_time
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(f"epoch {epoch:5d}/{args.epochs} | train {train_mse:.6f} | "
                      f"val {val_mse:.6f} | best {best_val_mse:.6f} "
                      f"| eta {int(eta // 60)}m{int(eta % 60)}s", flush=True)

    print(f"done in {int((timer.time() - start_time) / 60)} min | "
          f"best val {best_val_mse:.6f} | ckpt {checkpoint_path}")

    figure, axes = plt.subplots(figsize=(8, 5))
    for split, values in history.items():
        axes.semilogy(values, label=split, alpha=0.8)
    axes.set_xlabel("epoch"); axes.set_ylabel("MSE"); axes.set_title(experiment_name)
    axes.legend(); axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(pred_dir, "loss.png"), dpi=150)
    plt.close(figure)
    print(f"loss log + curve -> {pred_dir}/loss.txt , {pred_dir}/loss.png")


# ═══════════════════════════ inference ═══════════════════════════
# 3D scatter renders run in a process pool; the shared xyz is passed once via the
# pool initializer (it is too large to re-pickle per task).
_WORKER_XYZ = None


def _init_plot_worker(xyz):
    global _WORKER_XYZ
    _WORKER_XYZ = xyz
    import matplotlib
    matplotlib.use("Agg")


def _render_scatter_svg(task):
    values, cmap, vmin, vmax, out_path = task
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6)); fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d"); ax.patch.set_alpha(0.0)
    ax.scatter(_WORKER_XYZ[:, 0], _WORKER_XYZ[:, 1], _WORKER_XYZ[:, 2],
               c=values, cmap=cmap, s=1, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_axis_off(); plt.tight_layout()
    plt.savefig(out_path, format="svg", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def save_colorbar(cmap, vmin, vmax, label, out_path, half=False):
    """Standalone vertical colorbar SVG (inherited from Geo_DONet/main.py)."""
    fig, ax = plt.subplots(figsize=(1.2, 2.5 if half else 5))
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(sm, cax=ax)
    cb.set_label(label, fontsize=15)
    from matplotlib.ticker import MaxNLocator
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    cb.set_ticks(ticks[(ticks >= vmin) & (ticks <= vmax)])
    cb.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close()


def save_vm_traces(prediction, truth, time_ms, out_path, n_sample_nodes=5):
    """V_m(t) at a few evenly-spaced nodes. `truth` may be None (pred only)."""
    nodes = np.linspace(0, prediction.shape[0] - 1, n_sample_nodes, dtype=int)
    fig, axes = plt.subplots(len(nodes), 1, figsize=(10, 2.2 * len(nodes)), sharex=True)
    for ax, node in zip(axes, nodes):
        if truth is not None:
            ax.plot(time_ms, truth[node], "k-", lw=2.5, label="GT")
        ax.plot(time_ms, prediction[node], "r--", lw=2.0, label="Pred")
        ax.set_ylabel(f"node {node}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close()


def _print_metric_table(name, case_names, col1, col2, head1, head2):
    n_cases = len(col1)
    print(f"\n{name + ' metrics':<34}{head1:>10}{head2:>12}")
    print("-" * 56)
    for c in range(n_cases):
        label = str(case_names[c]) if case_names is not None else f"case{c}"
        print(f"{label[:34]:<34}{col1[c]:>10.4f}{col2[c]:>12.2f}")
    print("-" * 56)
    print(f"{'mean':<34}{np.nanmean(col1):>10.4f}{np.nanmean(col2):>12.2f}")
    print(f"{'std':<34}{np.nanstd(col1):>10.4f}{np.nanstd(col2):>12.2f}")


def _print_metric_summary(n_cases, cases_name, vm_rel, vm_mae, at_rel, at_mae):
    """Compact mean ± std block — the headline result, easy to paste into slides."""
    def stat(values): return np.nanmean(values), np.nanstd(values)
    vr, vmae, ar, amae = stat(vm_rel), stat(vm_mae), stat(at_rel), stat(at_mae)
    print(f"\nresults — {n_cases} cases ({cases_name}):")
    print(f"  {'V_m':<4}Rel L2  {vr[0]:.4f} ± {vr[1]:.4f}      MAE  {vmae[0]:.2f} ± {vmae[1]:.2f} mV")
    print(f"  {'AT':<4}Rel L2  {ar[0]:.4f} ± {ar[1]:.4f}      MAE  {amae[0]:.2f} ± {amae[1]:.2f} ms")


def _render_snapshots(out_dir, xyz, pred, truth, at_pred, at_true, time_ms, labels):
    """3D V_m field snapshots (a few frames) + AT maps for the first len(labels) cases."""
    snap_dir = os.path.join(out_dir, "snapshots"); os.makedirs(snap_dir, exist_ok=True)
    at_dir = os.path.join(out_dir, "activation_time"); os.makedirs(at_dir, exist_ok=True)
    targets = np.arange(0, min(300.0, time_ms[-1]) + 1, 10)       # ~10 ms steps
    frame_idx = [int(np.argmin(np.abs(time_ms - t))) for t in targets]

    tasks = []
    for c, label in enumerate(labels):
        for fi in frame_idx:
            tag = f"{label}_t{time_ms[fi]:03.0f}ms"
            tasks.append((pred[c, :, fi], CMAP_VM, VM_MIN, VM_MAX,
                          os.path.join(snap_dir, f"{tag}_Pred.svg")))
            if truth is not None:
                tasks.append((truth[c, :, fi], CMAP_VM, VM_MIN, VM_MAX,
                              os.path.join(snap_dir, f"{tag}_GT.svg")))
                tasks.append((np.abs(truth[c, :, fi] - pred[c, :, fi]),
                              CMAP_ERR, VM_ERR_MIN, VM_ERR_MAX,
                              os.path.join(snap_dir, f"{tag}_AbsErr.svg")))
        # activation-time maps (shared GT+Pred colour range when GT is present)
        ap = at_pred[c]
        if truth is not None and at_true is not None:
            at = at_true[c]
            lo, hi = float(np.nanmin([ap, at])), float(np.nanmax([ap, at]))
            tasks.append((ap, CMAP_AT, lo, hi, os.path.join(at_dir, f"{label}_AT_Pred.svg")))
            tasks.append((at, CMAP_AT, lo, hi, os.path.join(at_dir, f"{label}_AT_GT.svg")))
            tasks.append((np.abs(at - ap), CMAP_ERR, 0.0, float(np.nanmax(np.abs(at - ap))),
                          os.path.join(at_dir, f"{label}_AT_AbsErr.svg")))
        else:
            tasks.append((ap, CMAP_AT, float(np.nanmin(ap)), float(np.nanmax(ap)),
                          os.path.join(at_dir, f"{label}_AT_Pred.svg")))

    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_plot_worker,
                             initargs=(xyz,)) as pool:
        list(pool.map(_render_scatter_svg, tasks))
    print(f"  rendered {len(tasks)} scatter SVGs -> {snap_dir} , {at_dir}")


def _save_colorbars(out_dir, at_pred, at_true):
    cbar_dir = os.path.join(out_dir, "colorbars"); os.makedirs(cbar_dir, exist_ok=True)
    save_colorbar(CMAP_VM, VM_MIN, VM_MAX, "V_m (mV)",
                  os.path.join(cbar_dir, "cbar_Vm.svg"))
    save_colorbar(CMAP_ERR, VM_ERR_MIN, VM_ERR_MAX, "|dV_m| (mV)",
                  os.path.join(cbar_dir, "cbar_Vm_AbsErr.svg"), half=True)
    if at_pred is not None:
        vals = at_pred if at_true is None else np.concatenate([at_pred.ravel(), at_true.ravel()])
        save_colorbar(CMAP_AT, float(np.nanmin(vals)), float(np.nanmax(vals)), "AT (ms)",
                      os.path.join(cbar_dir, "cbar_AT.svg"))
        if at_true is not None:
            save_colorbar(CMAP_ERR, 0.0, float(np.nanmax(np.abs(at_pred - at_true))),
                          "|dAT| (ms)", os.path.join(cbar_dir, "cbar_AT_AbsErr.svg"), half=True)
    print(f"  saved colorbars -> {cbar_dir}")


def write_vtu_series(case_dir, points, tetra, time_ms, fields):
    """Per-frame .vtu + a Vm.pvd ParaView collection. `fields` maps name -> (n_nodes,
    n_frames) array; each frame writes that column as point_data. Modeled on
    scratch_scripts/visualize_neighbour_diff.py."""
    series_dir = os.path.join(case_dir, "series"); os.makedirs(series_dir, exist_ok=True)
    cells = [("tetra", tetra)]
    rows = []
    for t in range(time_ms.shape[0]):
        frame_file = f"frame_{t:04d}.vtu"
        meshio.write_points_cells(os.path.join(series_dir, frame_file), points, cells,
                                  point_data={name: values[:, t] for name, values in fields.items()})
        rows.append(f'    <DataSet timestep="{float(time_ms[t])}" group="" part="0" '
                    f'file="series/{frame_file}"/>')
    pvd = (['<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>'] + rows + ['  </Collection>', '</VTKFile>'])
    with open(os.path.join(case_dir, "Vm.pvd"), "w") as f:
        f.write("\n".join(pvd))


def fit_normalizer(npz_dict, n_train):
    """Build the (normalizer, reference) from a training npz — fit on the first
    n_train cases. Rebuilt every run (checkpoints store only weights), so the
    --train-data passed here must match what the model was trained on."""
    if npz_dict["vm"] is None or npz_dict["time"] is None:
        raise SystemExit("normalizer needs a training npz carrying both 'vm' and 'time' "
                         "— pass --train-data <training npz>")
    idx = np.arange(n_train)
    normalizer = Normalizer(npz_dict["theta"][idx], npz_dict["vm"][idx],
                            npz_dict["coords"], npz_dict["time"])
    return normalizer, {"coords": npz_dict["coords"], "time": npz_dict["time"]}


def resolve_cases(cases, n_total, n_train, n_val, default_cases):
    """Index array + label of the cases to run on. `cases` may be None (-> default_cases),
    a split name (all/train/val/test), or explicit global indices ('100' or '100,101')."""
    cases = cases or default_cases
    if cases == "all":
        return np.arange(n_total), "all"
    if cases in ("train", "val", "test"):
        train_idx, val_idx, test_idx = split_indices(n_total, n_train, n_val)
        selected = {"train": train_idx, "val": val_idx, "test": test_idx}[cases]
        if len(selected) == 0:
            raise SystemExit(f"--cases {cases}: empty for {n_total} cases (n_train={n_train}, "
                             f"n_val={n_val}). If --data-path is already a subset, pass --cases all.")
        return selected, cases
    try:
        idx = np.array([int(token) for token in cases.split(",")], dtype=int)
    except ValueError:
        raise SystemExit(f"--cases must be all|train|val|test or comma-separated indices, "
                         f"got {cases!r}")
    if idx.min() < 0 or idx.max() >= n_total:
        raise SystemExit(f"--cases index out of range for {n_total} cases: {cases}")
    return idx, f"idx[{cases}]"


def load_model(model_path, device):
    """One load path for any checkpoint: read weights, remap legacy keys if needed,
    infer the architecture from the weight shapes, build + load. Returns (model, config)."""
    raw = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
    if is_legacy_state_dict(state_dict):
        state_dict = remap_legacy_state_dict(state_dict)
    config = config_from_state_dict(state_dict)
    model = build_model(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def infer(args, device, test_mode):
    """test_mode=True (--test-model): test split + evaluate. test_mode=False (--infer):
    all cases; evaluate only if --eval. Outputs are opt-in (--snapshot/--vtu-out/
    --color-bar/--save); metrics always print when evaluating."""
    if not args.model_path:
        raise SystemExit("inference requires --model-path PATH")
    do_eval = test_mode or args.eval
    default_cases = "test" if test_mode else "all"

    model, config = load_model(args.model_path, device)
    print(f"model: GeoDONet {config} | {device}")

    # input cases to predict on (theta + Cobiveco required; V_m / time optional)
    data_path = resolve_data_path(args.data_file)
    inputs = load_inputs(data_path)

    # normalizer + reference rebuilt from the training npz every run (reuse --data-path
    # when it is already that npz with GT, else load --train-data)
    train_path = resolve_data_path(args.train_data)
    train_npz = (inputs if train_path == data_path and inputs["vm"] is not None
                 else load_inputs(train_path))
    normalizer, reference = fit_normalizer(train_npz, args.n_train)
    reference_time = np.asarray(reference["time"], dtype=np.float32)
    print(f"normalizer rebuilt from {os.path.basename(train_path)} (first {args.n_train} cases)")

    # ── select cases ──
    time_ms = inputs["time"] if inputs["time"] is not None else reference_time
    selected, cases_name = resolve_cases(args.cases, inputs["theta"].shape[0],
                                         args.n_train, args.n_val, default_cases)
    theta, coords = inputs["theta"][selected], inputs["coords"]
    truth = inputs["vm"][selected] if inputs["vm"] is not None else None
    case_names = inputs["case_names"][selected] if inputs["case_names"] is not None else None
    n_cases, n_nodes = theta.shape[0], coords.shape[0]
    labels = [str(case_names[c]) if case_names is not None else f"case{int(selected[c])}"
              for c in range(n_cases)]
    print(f"{'test-model' if test_mode else 'infer'}: {n_cases} cases ({cases_name}), "
          f"{n_nodes} nodes, {len(time_ms)} frames | "
          f"GT {'present' if truth is not None else 'absent'} | {device}")
    if do_eval and truth is None:
        raise SystemExit("evaluation needs ground-truth V_m in --data-path "
                         "(--test-model always evaluates; for --infer, drop --eval)")

    # ── predict per case (bounded memory), denormalize to physical mV ──
    to_device = lambda array: torch.tensor(array, dtype=torch.float32, device=device)
    trunk_chunks = build_trunk_chunks(to_device(normalizer.coords(coords)),
                                      to_device(normalizer.time(time_ms)), CHUNK_FRAMES)
    theta_norm = to_device(normalizer.theta(theta))
    with torch.no_grad():
        pred = np.stack([
            to_numpy(chunked_forward(model, theta_norm[i:i + 1], trunk_chunks, n_nodes)[0])
            for i in range(n_cases)])
    pred = normalizer.vm_inverse(pred).astype(np.float32)        # (n_cases, n_nodes, n_frames)

    out_dir = args.out_dir or os.path.join("Predictions", checkpoint_stem(args.model_path))

    # ── evaluation (metrics always print; arrays go into predictions.npz if --save) ──
    at_pred = at_true = None
    metrics = {}
    if do_eval:
        vm_rel, vm_mae = vm_rel_l2_mae(pred, truth)
        at_pred = np.stack([activation_time(pred[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
        at_true = np.stack([activation_time(truth[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
        at_rel, at_mae = at_rel_l2_mae(at_pred, at_true)
        _print_metric_table("V_m", case_names, vm_rel, vm_mae, "Rel L2", "MAE (mV)")
        _print_metric_table("AT", case_names, at_rel, at_mae, "Rel L2", "MAE (ms)")
        _print_metric_summary(n_cases, cases_name, vm_rel, vm_mae, at_rel, at_mae)
        metrics = dict(vm_rel_l2=vm_rel, vm_mae=vm_mae, at_pred=at_pred, at_true=at_true,
                       at_rel_l2=at_rel, at_mae=at_mae)

    # ── --snapshot: V_m(t) traces (xyz-free) + 3D V_m/AT scatter SVGs (needs --reference-xyz) ──
    if args.snapshot:
        n_viz = min(2, n_cases)
        traces_dir = os.path.join(out_dir, "traces"); os.makedirs(traces_dir, exist_ok=True)
        for c in range(n_viz):
            save_vm_traces(pred[c], None if truth is None else truth[c], time_ms,
                           os.path.join(traces_dir, f"{labels[c]}_traces.png"))
        print(f"  traces -> {traces_dir}")
        if os.path.exists(args.reference_xyz):
            xyz = np.load(args.reference_xyz)["cartesian_coords"].astype(np.float32)
            if xyz.shape[0] != n_nodes:
                raise SystemExit(f"reference xyz has {xyz.shape[0]} nodes != {n_nodes} predicted")
            if at_pred is None:        # AT not computed in the eval block (e.g. --infer w/o --eval)
                at_pred = np.stack([activation_time(pred[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
                if truth is not None:
                    at_true = np.stack([activation_time(truth[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
            _render_snapshots(out_dir, xyz, pred, truth, at_pred, at_true, time_ms, labels[:n_viz])
        else:
            print(f"  3D scatter skipped (--reference-xyz not found: {args.reference_xyz})")

    # ── --vtu-out: predicted V_m .vtu+.pvd series per case (+ GT/abserr fields when GT present) ──
    if args.vtu_out:
        if not os.path.exists(args.mesh):
            raise SystemExit(f"--vtu-out needs a reference mesh (--mesh); not found: {args.mesh}")
        mesh = meshio.read(args.mesh)
        points = mesh.points.astype(np.float32)
        tetra = next((cb.data for cb in mesh.cells if cb.type == "tetra"), None)
        if tetra is None:
            raise SystemExit(f"--mesh {args.mesh} has no tetra cells")
        if points.shape[0] != n_nodes:
            raise SystemExit(f"--mesh has {points.shape[0]} points != {n_nodes} predicted nodes")
        tetra = tetra.astype(np.int32)
        vtu_root = os.path.join(out_dir, "vtu"); os.makedirs(vtu_root, exist_ok=True)
        for c in range(n_cases):
            fields = {"Vm": pred[c]}
            if truth is not None:
                fields["Vm_gt"] = truth[c]
                fields["Vm_abserr"] = np.abs(truth[c] - pred[c]).astype(np.float32)
            write_vtu_series(os.path.join(vtu_root, labels[c]), points, tetra, time_ms, fields)
        print(f"  wrote {n_cases} V_m .vtu series ({len(time_ms)} frames each) -> {vtu_root}")

    # ── --color-bar ──
    if args.color_bar:
        os.makedirs(out_dir, exist_ok=True)
        _save_colorbars(out_dir, at_pred, at_true)

    # ── --save: full predictions.npz ──
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
        saved = {"pred": pred, "time": time_ms, "case_names": np.asarray(labels)}
        if truth is not None:
            saved["true"] = truth
        saved.update(metrics)
        np.savez_compressed(os.path.join(out_dir, "predictions.npz"), **saved)
        print(f"  saved predictions.npz -> {out_dir}")

    wrote_files = args.save or args.snapshot or args.vtu_out or args.color_bar
    print(f"inference complete" + (f" -> {out_dir}" if wrote_files
                                   else " (metrics only; no files written)"))


# ══════════════════════════════ main ══════════════════════════════
def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.test_model:
        infer(args, device, test_mode=True)
    elif args.infer:
        infer(args, device, test_mode=False)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
