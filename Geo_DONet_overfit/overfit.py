"""
Single-case overfit — capacity test for the V_m depolarization upshoot.

The clean Geo-DONet reproduces the bulk V_m field but smears the sharp ~1-2 ms
depolarization upstroke. Hypothesis: spectral bias — the smooth Tanh trunk (a
coordinate-MLP) cannot represent the sharp space-time wavefront.

This script isolates *capacity* from *generalization* by overfitting the network
on a single heart. With one geometry the branch `branch(theta_0)` is a constant
vector b, so the prediction

    V_m(x, t) = <b, trunk(x, t)>

collapses to a coordinate-MLP (the trunk) with a learned linear head. Driving the
train MSE toward zero on one case therefore measures exactly what the trunk can
represent — no train/val gap to confound it.

  - Fits the upshoot   -> capacity is there; the deployed failure is generalization.
  - Still smears it     -> spectral bias confirmed (at f601 the 1 ms grid resolves
                           the upstroke, so a residual smear is the network, not sampling).

It imports the *actual* architecture and helpers from the sibling Geo_DONet/ so it
tests the same network, not a copy. Run on a compute node (full grid forward each
epoch); never the login node.

Usage:
    python overfit.py                                   # case 0, f601 npz @ frame-step 1
    python overfit.py --data-path geo_donet_data_f601.npz --frame-step 5   # f121 (5 ms)
    python overfit.py --frame-step 1 --grad-checkpoint  # f601 (1 ms), memory-bounded
"""
import os
import sys
import argparse
import time as timer

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR

# reuse the *actual* clean Geo_DONet architecture + helpers (single source of truth)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Geo_DONet"))
from opnn import GeoDONet, build_trunk_chunks, chunked_forward        # noqa: E402
from utils import Normalizer, mse, activation_time, to_numpy, load_dataset  # noqa: E402


# ════════════════════════ CONFIGURATION ════════════════════════
# Defaults for a run; each is exposed as a CLI flag (below) that overrides it.
DATA_BASE = "/home/svu/e1032484/scratch"
DATA_FILE = "geo_donet_data_f601.npz"   # the f601 npz; --frame-step 5 -> f121, 1 -> f601
FRAME_STEP = 1                          # stride the time axis on load (1 -> f601, 5 -> f121)
CASE = 0                                # which heart to overfit

WIDTH, DEPTH = 300, 4                   # baseline architecture (the one under test)
LR = 5e-4
LR_SCHEDULED = False                    # linearly decay LR to 0.1x over all epochs
EPOCHS = 5000                           # cap; early stopping usually halts a single-case overfit sooner
PATIENCE = 500                          # early stop: epochs with no >MIN_DELTA train-MSE improvement (0 = off)
MIN_DELTA = 1e-4                        # relative train-MSE improvement that counts as progress
GRAD_CHECKPOINT = False                 # recompute trunk in backward (needed for f301/f601 memory)
SEED = 42
DEVICE = None                           # None -> auto (cuda if available)

CHUNK_FRAMES = 25                       # trunk frames per forward chunk
AT_THRESHOLD_MV = -10.0                 # activation = first upward crossing of this level
EVAL_EVERY = 50                         # epochs between progress prints + loss.txt flush (diagnostics run at end)


def parse_args():
    p = argparse.ArgumentParser(description="Single-case overfit capacity test (Geo-DONet)")
    p.add_argument("--data-path", dest="data_file", type=str, default=None, metavar="PATH",
                   help=f"npz: absolute path or bare name under DATA_BASE (default {DATA_FILE})")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP, metavar="K",
                   help="stride the npz time axis by K (1 -> f601, 5 -> f121). default 1")
    p.add_argument("--case", type=int, default=CASE, help="case index to overfit")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-scheduled", action=argparse.BooleanOptionalAction, default=LR_SCHEDULED,
                   help="linearly decay LR from --lr to 0.1x over all epochs")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="max epochs (early stopping may halt sooner)")
    p.add_argument("--patience", type=int, default=PATIENCE,
                   help="early stop after this many epochs with no >--min-delta (relative) "
                        "train-MSE improvement (0 = off)")
    p.add_argument("--min-delta", type=float, default=MIN_DELTA,
                   help="relative train-MSE improvement that resets the early-stop patience")
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=GRAD_CHECKPOINT,
                   help="recompute the trunk in backward to bound memory (needed for f301/f601)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                   help="use TF32 matmul on Ampere+ (3-5x faster; ~1e-3 relative precision "
                        "floor). --no-tf32 for a full-fp32 run if you need MSE below that.")
    p.add_argument("--device", type=str, default=DEVICE, help="cuda/cpu (default: auto)")
    p.add_argument("--out-dir", type=str, default=None, help="override the output dir")
    return p.parse_args()


def resolve_data_path(data_file):
    if data_file is None:
        return os.path.join(DATA_BASE, DATA_FILE)
    return data_file if os.path.isabs(data_file) else os.path.join(DATA_BASE, data_file)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subsample_time(vm, time, frame_step):
    """Stride the time axis (vm last axis + time) by frame_step. <=1 is a no-op."""
    if frame_step <= 1:
        return vm, time
    return vm[..., ::frame_step], time[::frame_step]


# ════════════════════════════ training ════════════════════════════
def train(args, device, out_dir, data):
    def to_device(array):
        return torch.tensor(array, dtype=torch.float32, device=device)

    # ── data: keep only the one case we overfit ──
    vm_all, time = subsample_time(data["vm"], data["time"], args.frame_step)
    if not (0 <= args.case < vm_all.shape[0]):
        raise SystemExit(f"--case {args.case} out of range (0..{vm_all.shape[0] - 1})")
    theta = data["theta"][args.case:args.case + 1]        # (1, geo_dim) — one geometry
    coords = data["coords"]                               # (n_nodes, coord_dim) — shared mesh
    vm = vm_all[args.case:args.case + 1]                  # (1, n_nodes, n_frames)
    n_nodes, n_frames = vm.shape[1], vm.shape[2]
    case_label = (str(data["case_names"][args.case]) if data["case_names"] is not None
                  else f"case{args.case}")
    dt = float(time[1] - time[0]) if n_frames > 1 else 0.0
    print(f"overfitting {case_label} (frame-step {args.frame_step}) "
          f"| {n_nodes} nodes, {n_frames} frames, dt={dt:g} ms", flush=True)

    # normalize on this single case (self-consistent for an overfit; the theta-std
    # guard in Normalizer handles the single-sample std=0).
    normalizer = Normalizer(theta, vm, coords, time)
    theta_norm = to_device(normalizer.theta(theta))                  # (1, geo_dim)
    vm_norm = to_device(normalizer.vm(vm))                           # (1, n_nodes, n_frames)
    trunk_chunks = build_trunk_chunks(to_device(normalizer.coords(coords)),
                                      to_device(normalizer.time(time)), CHUNK_FRAMES)

    # ── model ──
    model = GeoDONet(geo_dim=theta.shape[1], coord_dim=coords.shape[1],
                     width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, 1.0, 0.1, args.epochs) if args.lr_scheduled else None
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"model: GeoDONet w{args.width} d{args.depth} | {n_parameters:,} params | {device}", flush=True)

    # ── train: full-batch GD on the one case; loss.txt written incrementally ──
    history = []
    best_mse = float("inf")
    epochs_since_improve = 0
    best_path = os.path.join(out_dir, "best.pt")
    start_time = timer.time()
    with open(os.path.join(out_dir, "loss.txt"), "w") as loss_log:
        loss_log.write("# epoch\ttrain_mse\n")
        for epoch in range(args.epochs):
            model.train()
            pred = chunked_forward(model, theta_norm, trunk_chunks, n_nodes,
                                   use_checkpoint=args.grad_checkpoint)
            loss = mse(pred, vm_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_mse = loss.item()
            history.append(train_mse)
            loss_log.write(f"{epoch}\t{train_mse:.8e}\n")

            # best.pt always tracks the strictly-best weights; patience resets only on a
            # *meaningful* (> min_delta relative) improvement, so a slow asymptote stops it.
            if train_mse < best_mse:
                if train_mse < best_mse * (1 - args.min_delta):
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                best_mse = train_mse
                torch.save({"model_state_dict": model.state_dict()}, best_path)
            else:
                epochs_since_improve += 1

            if epoch % EVAL_EVERY == 0 or epoch == args.epochs - 1:
                loss_log.flush()
                elapsed = timer.time() - start_time
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(f"epoch {epoch:6d}/{args.epochs} | train MSE {train_mse:.3e} "
                      f"| best {best_mse:.3e} | eta {int(eta // 60)}m{int(eta % 60)}s", flush=True)

            if args.patience > 0 and epochs_since_improve >= args.patience:
                loss_log.flush()
                print(f"early stop at epoch {epoch}: no >{args.min_delta:.0e} relative "
                      f"train-MSE improvement for {args.patience} epochs (best {best_mse:.3e})",
                      flush=True)
                break

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(out_dir, "last.pt"))
    print(f"done in {(timer.time() - start_time) / 60:.1f} min | best train MSE {best_mse:.3e}", flush=True)

    # final prediction (best weights) in physical mV for diagnostics
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_norm = chunked_forward(model, theta_norm, trunk_chunks, n_nodes)
    pred = normalizer.vm_inverse(to_numpy(pred_norm[0])).astype(np.float32)  # (n_nodes, n_frames)
    truth = vm[0]                                                            # (n_nodes, n_frames)
    return history, pred, truth, np.asarray(time, np.float32), case_label


# ═══════════════════════════ diagnostics ═══════════════════════════
def loss_curve(history, out_dir, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history, color="tab:blue")
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE (normalized V_m)")
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150); plt.close(fig)


def per_frame_error(pred, truth, time_ms, out_dir):
    """Rel L2 over nodes at each frame. A spike at the depolarization frames is the
    upshoot failure. Returns the array so the summary can quote its peak."""
    num = np.linalg.norm(pred - truth, axis=0)               # (n_frames,)
    den = np.linalg.norm(truth, axis=0) + 1e-12
    rel = num / den
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_ms, rel, color="tab:red")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("per-frame Rel L2 (over nodes)")
    ax.set_title("Where the error lives in time"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "frame_error.png"), dpi=150); plt.close(fig)
    return rel


def pick_trace_nodes(at_true, k_random=2, rng=None):
    """Nodes spanning activation time: earliest, 25/50/75 pct, latest, + a couple random."""
    activated = np.where(np.isfinite(at_true))[0]
    if len(activated) == 0:
        return np.arange(min(6, at_true.shape[0]))
    order = activated[np.argsort(at_true[activated])]
    picks = [order[0], order[len(order) // 4], order[len(order) // 2],
             order[3 * len(order) // 4], order[-1]]
    rng = rng or np.random.default_rng(0)
    picks += list(rng.choice(activated, size=min(k_random, len(activated)), replace=False))
    return np.array(sorted(set(int(n) for n in picks)))


def node_traces(pred, truth, time_ms, nodes, at_true, out_dir):
    """Overlay GT vs predicted V_m(t) for representative nodes — the money plot."""
    n = len(nodes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for i, node in enumerate(nodes):
        ax = axes[i // cols][i % cols]
        ax.plot(time_ms, truth[node], color="black", lw=2, label="GT")
        ax.plot(time_ms, pred[node], color="tab:orange", lw=1.5, ls="--", label="pred")
        at = at_true[node]
        title = f"node {node}" + (f" (AT {at:.0f} ms)" if np.isfinite(at) else " (no AT)")
        ax.set_title(title); ax.set_xlabel("time (ms)"); ax.set_ylabel("V_m (mV)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "traces.png"), dpi=150); plt.close(fig)


def at_scatter(at_pred, at_true, out_dir):
    """pred vs true activation time over nodes activating in both. Returns AT MAE (ms)."""
    valid = np.isfinite(at_pred) & np.isfinite(at_true)
    if not valid.any():
        return float("nan"), 0
    ap, at = at_pred[valid], at_true[valid]
    mae = float(np.abs(ap - at).mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(at, ap, s=3, alpha=0.3, color="tab:blue")
    lim = [float(min(at.min(), ap.min())), float(max(at.max(), ap.max()))]
    ax.plot(lim, lim, color="black", lw=1, ls="--")
    ax.set_xlabel("true AT (ms)"); ax.set_ylabel("pred AT (ms)")
    ax.set_title(f"Activation time | MAE {mae:.2f} ms (n={valid.sum()})")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "at_scatter.png"), dpi=150); plt.close(fig)
    return mae, int(valid.sum())


def upstroke_steepness(pred, truth, time_ms, out_dir):
    """Max dV/dt per node (mV/ms): how much the predicted rise is flattened vs GT."""
    dt = np.diff(time_ms)
    slope_pred = (np.diff(pred, axis=1) / dt).max(axis=1)
    slope_true = (np.diff(truth, axis=1) / dt).max(axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(slope_true, slope_pred, s=3, alpha=0.3, color="tab:green")
    lim = [0.0, float(slope_true.max() * 1.05)]
    ax.plot(lim, lim, color="black", lw=1, ls="--")
    ax.set_xlabel("true max dV/dt (mV/ms)"); ax.set_ylabel("pred max dV/dt (mV/ms)")
    ax.set_title("Upstroke steepness (points below diagonal = flattened)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "upstroke.png"), dpi=150); plt.close(fig)
    return float(np.median(slope_true)), float(np.median(slope_pred))


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32      # Ampere tensor cores for fp32 matmul
    torch.backends.cudnn.allow_tf32 = args.tf32
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load the npz once, derive the output dir from the resulting frame count
    data_path = resolve_data_path(args.data_file)
    if not os.path.exists(data_path):
        raise SystemExit(f"data not found: {data_path}")
    data = load_dataset(data_path)
    n_frames = data["time"][::args.frame_step].shape[0] if args.frame_step > 1 else data["time"].shape[0]
    out_dir = args.out_dir or os.path.join(
        "Predictions", f"case{args.case}_f{n_frames}_w{args.width}_d{args.depth}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"outputs -> {out_dir}/", flush=True)

    history, pred, truth, time_ms, case_label = train(args, device, out_dir, data)

    # ── diagnostics ──
    loss_curve(history, out_dir, f"{case_label} | f{len(time_ms)} | w{args.width} d{args.depth}")
    rel = per_frame_error(pred, truth, time_ms, out_dir)
    at_true = activation_time(truth, time_ms, AT_THRESHOLD_MV)
    at_pred = activation_time(pred, time_ms, AT_THRESHOLD_MV)
    nodes = pick_trace_nodes(at_true)
    node_traces(pred, truth, time_ms, nodes, at_true, out_dir)
    at_mae, n_at = at_scatter(at_pred, at_true, out_dir)
    med_true, med_pred = upstroke_steepness(pred, truth, time_ms, out_dir)

    # ── summary -> overfit_log.txt ──
    overall_rel = float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + 1e-12))
    peak_frame = int(np.argmax(rel))
    lines = [
        f"overfit summary — {case_label} | f{len(time_ms)} | w{args.width} d{args.depth}",
        f"  final train MSE (normalized)   : {history[-1]:.3e}  (best {min(history):.3e})",
        f"  overall V_m Rel L2 (phys mV)   : {overall_rel:.4f}",
        f"  per-frame Rel L2 peak          : {rel[peak_frame]:.4f} at t={time_ms[peak_frame]:.0f} ms",
        f"  activation-time MAE            : {at_mae:.2f} ms  (over {n_at} nodes active in both)",
        f"  median max dV/dt true vs pred  : {med_true:.2f} vs {med_pred:.2f} mV/ms "
        f"({100 * med_pred / (med_true + 1e-12):.0f}% of GT steepness)",
        "  files: loss.png frame_error.png traces.png at_scatter.png upstroke.png",
    ]
    summary = "\n".join(lines)
    print(summary, flush=True)
    with open(os.path.join(out_dir, "overfit_log.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"diagnostics -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
