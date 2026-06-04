"""
Geo-DONet (SIREN trunk) — train / test / infer entry point.

Same clean structure and CLI as ../Geo_DONet, but the trunk is a SIREN (sine
activations) instead of Tanh, to fit the sharp V_m depolarisation upstroke. Adds
two SIREN-specific knobs over the baseline:
  - --omega-0       trunk frequency bandwidth (default 30, the SIREN paper recipe)
  - --ndiff-weight  signed-Laplacian neighbour-difference loss weight (0 = MSE only)

Functional modules (self-contained copy of the baseline's, so this folder stands alone):
  - architecture : opnn.GeoDONetSIREN / build_trunk_chunks / chunked_forward
  - data         : utils.load_dataset / load_inputs / split_indices / Normalizer
  - evaluation   : utils.mse (train) + vm_rel_l2_mae / activation_time, + ndiff loss
  - output       : viz.* (metric tables, traces, 3D scatters, colorbars, .vtu)

Every run default lives in the CONFIGURATION block below; each is exposed as a CLI
flag that overrides it. Run main.py from the folder it lives in — outputs are
written relative to the cwd (CheckPts/, Predictions/<experiment_name>/).

main() dispatches three modes (default = train):
  - (no flag)     train()                 train + validate; save the best-val weights
  - --test-model  infer(test_mode=True)   forward pass on the test split + evaluate vs GT
  - --infer       infer(test_mode=False)  forward pass on all cases; --eval to also score

Checkpoints store the model weights AND a `config` block — unlike the shape-only
GeoDONet checkpoints, because omega_0 is a forward-time multiplier, not a weight, so
it can't be inferred from the state_dict. The architecture is taken from that config
(or inferred for legacy original-opnn SIREN checkpoints, with omega_0 defaulting to
30, what they were trained at); the normalizer is rebuilt from the training npz
(--train-data) every run.

Usage:
    python main.py                                                  # train (defaults)
    python main.py --epochs 5000 --lr-scheduled --omega-0 30        # train, overrides
    python main.py --ndiff-weight 1.0                               # train + ndiff loss
    python main.py --test-model --model-path CheckPts/<run>.pt      # eval on the test split
    python main.py --test-model --model-path <legacy_siren>.pt      # re-eval an old opnn SIREN ckpt
    python main.py --infer --model-path <ckpt> --data-path new.npz --save   # predict, no GT
"""
import os
import argparse
import time as timer

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
from torch.optim.lr_scheduler import LinearLR

from utils import (load_dataset, load_inputs, split_indices, Normalizer,
                   mse, vm_rel_l2_mae, activation_time, at_rel_l2_mae, to_numpy,
                   build_laplacian_operator, signed_laplacian_loss)
from opnn import (GeoDONetSIREN, build_trunk_chunks, chunked_forward, build_model,
                  is_legacy_state_dict, remap_legacy_state_dict, config_from_state_dict,
                  LEGACY_OMEGA_0)
from viz import (format_metric_table, format_metric_summary, save_vm_traces,
                 render_snapshots, save_colorbars, write_vtu_series)


# ════════════════════════ CONFIGURATION ════════════════════════
# Defaults for every run. Each is exposed as a CLI flag (below) that overrides it.

# where the data lives — one absolute path; the npz, reference xyz (--snapshot) and
# reference vtu (--vtu-out) all resolve under it.
DATA_BASE = "/home/svu/e1032484/scratch"
DATA_FILE = "geo_donet_data_f601.npz"                  # default train/infer npz
FRAME_STEP = 5                                         # stride the time axis on load (2 -> f301 from f601, 5 -> f121)
REFERENCE_XYZ = os.path.join(DATA_BASE, "canonical_xyz.npz")  # xyz for --snapshot 3D scatter
REFERENCE_VTU = os.path.join(DATA_BASE, "canonical.vtu")      # mesh (cells) for --vtu-out

# train / val / test split (by case index order; rest after train+val = test)
N_TRAIN, N_VAL = 95, 5

# model architecture
WIDTH, DEPTH = 300, 4
OMEGA_0 = 30.0                                         # SIREN trunk frequency (paper recipe; lower=smoother)

# optimization. SIREN trains at a smaller LR than the Tanh baseline: default flat
# 1e-4; with --lr-scheduled it decays --lr -> 0.1x (pass --lr 5e-4 --lr-scheduled to
# reproduce the original 5e-4 -> 5e-5 schedule).
EPOCHS, BATCH_SIZE, LR, LR_SCHEDULED = 5000, 24, 1e-4, False
PATIENCE = 500                                         # early stop: epochs with no val improvement (0 = off)
GRAD_CHECKPOINT = True                                 # recompute trunk in backward — SIREN activations need it even at f121
SEED = 42

# ndiff auxiliary loss (signed-Laplacian neighbour difference; 0 = baseline MSE only)
NDIFF_WEIGHT = 0.0
ADJ_FILE = os.environ.get("DIMON_ADJ_FILE",
                          os.path.join(DATA_BASE, "reference", "mesh_adjacency_data_order.npz"))

DEVICE = None                                          # None -> auto (cuda if available)

# inference outputs — all opt-in. Evaluation is on for --test-model, or --eval for --infer.
SNAPSHOT = False                                       # V_m(t) traces + 3D V_m/AT scatter SVGs
VTU_OUT = False                                        # predicted V_m .vtu+.pvd series (needs --mesh)
N_VIZ = 2                                              # default cases rendered for --snapshot/--vtu-out
COLOR_BAR = False                                      # standalone colorbars
SAVE = False                                           # write predictions.npz

# fixed internals (rarely changed; no CLI flag)
CHUNK_FRAMES = 10                                      # trunk frames per forward chunk (smaller than Tanh's 25; SIREN memory)
AT_THRESHOLD_MV = -10.0                                # activation = first upward crossing


def parse_args():
    p = argparse.ArgumentParser(description="Geo-DONet (SIREN) — train / test / infer")

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
    p.add_argument("--patience", type=int, default=PATIENCE,
                   help="early stop after this many epochs with no val improvement (0 = off)")
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=GRAD_CHECKPOINT,
                   help="recompute the trunk in backward to bound GPU memory; ~1.3x slower, "
                        "identical result. On by default for SIREN (--no-grad-checkpoint to disable).")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--omega-0", type=float, default=OMEGA_0,
                   help="SIREN trunk frequency hyperparameter (default 30, the paper recipe for "
                        "images/SDFs). Lower -> smoother, closer to a Tanh net; higher -> sharper "
                        "features but harder to train. Stored in the checkpoint's config.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", type=str, default=DEVICE, help="cuda/cpu (default: auto)")
    p.add_argument("--model-path", type=str, default=None, metavar="PATH",
                   help="checkpoint path. train: output (default CheckPts/<auto-name>.pt); "
                        "test/infer: checkpoint to load (required)")

    # ── ndiff auxiliary loss (train only) ──
    p.add_argument("--ndiff-weight", type=float, default=NDIFF_WEIGHT,
                   help="weight lambda on the signed-Laplacian neighbour-difference loss "
                        "(0 = baseline MSE only). Total train loss = MSE + lambda * mean((L(pred-gt))^2), "
                        "L = I - D^-1 A.")
    p.add_argument("--adj-file", type=str, default=ADJ_FILE, metavar="PATH",
                   help="canonical-order mesh adjacency npz (edge_src/dst, n_nodes) for the ndiff loss")

    # ── data + split (shared by all modes) ──
    p.add_argument("--data-path", dest="data_file", type=str, default=None, metavar="PATH",
                   help=f"npz: absolute path or a bare name under DATA_BASE (default {DATA_FILE})")
    p.add_argument("--n-train", type=int, default=N_TRAIN, help="train split size")
    p.add_argument("--n-val", type=int, default=N_VAL, help="val split size (rest = test)")
    p.add_argument("--train-data", type=str, default=None, metavar="PATH",
                   help="npz the normalizer is rebuilt from each run, first --n-train cases "
                        "(default: the --data-path npz for --test-model, else DATA_FILE for --infer)")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP, metavar="K",
                   help="stride the npz time axis by K on load (2 -> f301 from the f601 npz, "
                        "5 -> f121). default 1 = all frames")

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
    p.add_argument("--n-viz", type=str, default=None, metavar="IDX|all",
                   help="cases to render for --snapshot/--vtu-out: comma-separated global case "
                        "indices (e.g. 116 or 100,116), or 'all'. default: first 2")
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
    leading 'model_chkpts_' (legacy naming). Drives Predictions/<stem>/."""
    base = os.path.splitext(os.path.basename(model_path))[0]
    return base[len("model_chkpts_"):] if base.startswith("model_chkpts_") else base


def subsample_frames(npz_dict, frame_step):
    """Stride the time axis (vm + time) of a loaded npz dict by frame_step; theta/coords
    are time-independent and untouched. frame_step <= 1 is a no-op. Lets the f601 npz
    stand in for f301 (K=2) or f121 (K=5) without a separate file."""
    if frame_step <= 1:
        return npz_dict
    out = dict(npz_dict)
    if out.get("time") is not None:
        out["time"] = out["time"][::frame_step]
    if out.get("vm") is not None:
        out["vm"] = out["vm"][..., ::frame_step]
    return out


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
    data = subsample_frames(load_dataset(data_path), args.frame_step)
    theta, coords, vm, time = data["theta"], data["coords"], data["vm"], data["time"]
    n_cases, n_nodes, n_frames = vm.shape
    train_idx, val_idx, test_idx = split_indices(n_cases, args.n_train, args.n_val)
    print(f"data: {data_path} (frame-step {args.frame_step})\n  {n_cases} cases, {n_nodes} nodes, "
          f"{n_frames} frames | {len(train_idx)} train / {len(val_idx)} val "
          f"({len(test_idx)} test held out)")

    # fit normalization on the train split only, then load just the splits we train on
    normalizer = Normalizer(theta[train_idx], vm[train_idx], coords, time)
    theta_train = to_device(normalizer.theta(theta[train_idx]))
    vm_train = to_device(normalizer.vm(vm[train_idx]))
    theta_val = to_device(normalizer.theta(theta[val_idx]))
    vm_val = to_device(normalizer.vm(vm[val_idx]))
    coords_norm = to_device(normalizer.coords(coords))      # (n_nodes, coord_dim)
    time_norm = to_device(normalizer.time(time))            # (n_frames,)
    trunk_chunks = build_trunk_chunks(coords_norm, time_norm, CHUNK_FRAMES)

    # ── model ─────────────────────────────────────────────────────────────
    model = GeoDONetSIREN(geo_dim=theta.shape[1], coord_dim=coords.shape[1],
                          width=args.width, depth=args.depth, omega_0=args.omega_0).to(device)
    model_config = {"geo_dim": int(theta.shape[1]), "coord_dim": int(coords.shape[1]),
                    "width": int(args.width), "depth": int(args.depth),
                    "omega_0": float(args.omega_0)}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, 1.0, 0.1, args.epochs) if args.lr_scheduled else None
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"model: GeoDONetSIREN w{args.width} d{args.depth} omega_0={args.omega_0:g} "
          f"| {n_parameters:,} params | {device}")
    print(f"LR: {args.lr:g} ({'scheduled -> 0.1x' if args.lr_scheduled else 'flat'}) | "
          f"chunk {CHUNK_FRAMES} frames | grad-checkpoint {args.grad_checkpoint}")

    # ndiff auxiliary loss (train only); off when --ndiff-weight 0
    ahat = None
    if args.ndiff_weight > 0:
        ahat = build_laplacian_operator(args.adj_file, n_nodes, device)
        print(f"ndiff loss ON: signed-Laplacian, weight={args.ndiff_weight:g}")
    else:
        print("ndiff loss OFF (--ndiff-weight 0): baseline MSE")

    # output paths: checkpoint (best-val weights) -> CheckPts/<name>.pt;
    # loss log + curve -> Predictions/<name>/.  experiment_name = checkpoint stem.
    default_name = f"geodonet_siren_w{args.width}_d{args.depth}_w0_{args.omega_0:g}_{args.epochs}ep"
    default_name += "_lrsched" if args.lr_scheduled else ""
    default_name += f"_ndiff{args.ndiff_weight:g}" if args.ndiff_weight > 0 else ""
    default_name += f"_f{n_frames}" if args.frame_step != 1 else ""
    default_name += "" if args.seed == SEED else f"_seed{args.seed}"
    checkpoint_path = args.model_path or os.path.join("CheckPts", f"{default_name}.pt")
    experiment_name = checkpoint_stem(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    pred_dir = os.path.join("Predictions", experiment_name)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"checkpoint -> {checkpoint_path} | logs -> {pred_dir}/")

    # ── train (loss log written incrementally so a wall-killed run keeps its log) ──
    history = {"train": [], "val": [], "ndiff": []}
    best_val_mse = float("inf")
    epochs_since_improve = 0
    train_order = np.arange(len(train_idx))
    start_time = timer.time()
    with open(os.path.join(pred_dir, "loss.txt"), "w") as loss_log:
        loss_log.write("# epoch\ttrain_mse\tval_mse\tndiff\n")
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(train_order)
            running_mse, running_ndiff, n_batches = 0.0, 0.0, 0
            for batch_start in range(0, len(train_order), args.batch_size):
                batch = train_order[batch_start:batch_start + args.batch_size]
                pred = chunked_forward(model, theta_train[batch], trunk_chunks, n_nodes,
                                       use_checkpoint=args.grad_checkpoint)
                err = pred - vm_train[batch]                 # (B, n_nodes, n_frames)
                mse_loss = (err ** 2).mean()
                if ahat is not None:
                    ndiff = signed_laplacian_loss(err, ahat)
                    loss = mse_loss + args.ndiff_weight * ndiff
                    running_ndiff += ndiff.item()
                else:
                    loss = mse_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_mse += mse_loss.item()
                n_batches += 1
            if scheduler:
                scheduler.step()
            train_mse = running_mse / n_batches
            ndiff_mean = running_ndiff / n_batches

            model.eval()
            with torch.no_grad():
                val_mse = mse(chunked_forward(model, theta_val, trunk_chunks, n_nodes),
                              vm_val).item()
            history["train"].append(train_mse)
            history["val"].append(val_mse)
            history["ndiff"].append(ndiff_mean)
            loss_log.write(f"{epoch}\t{train_mse:.8f}\t{val_mse:.8f}\t{ndiff_mean:.8f}\n")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                epochs_since_improve = 0
                torch.save({"model_state_dict": model.state_dict(), "config": model_config},
                           checkpoint_path)
            else:
                epochs_since_improve += 1

            if args.patience > 0 and epochs_since_improve >= args.patience:
                print(f"early stop at epoch {epoch}: no val improvement for "
                      f"{args.patience} epochs (best val {best_val_mse:.6f})", flush=True)
                break

            if epoch % 10 == 0:
                loss_log.flush()
                elapsed = timer.time() - start_time
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                ndiff_str = f" | ndiff {ndiff_mean:.6f}" if ahat is not None else ""
                print(f"epoch {epoch:5d}/{args.epochs} | train {train_mse:.6f} | "
                      f"val {val_mse:.6f} | best {best_val_mse:.6f}{ndiff_str} "
                      f"| eta {int(eta // 60)}m{int(eta % 60)}s", flush=True)

    print(f"done in {int((timer.time() - start_time) / 60)} min | "
          f"best val {best_val_mse:.6f} | ckpt {checkpoint_path}")

    figure, axes = plt.subplots(figsize=(8, 5))
    for split in ("train", "val"):
        axes.semilogy(history[split], label=split, alpha=0.8)
    if ahat is not None:
        axes.semilogy(history["ndiff"], label="ndiff", alpha=0.6, ls="--")
    axes.set_xlabel("epoch"); axes.set_ylabel("MSE"); axes.set_title(experiment_name)
    axes.legend(); axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(pred_dir, "loss.png"), dpi=150)
    plt.close(figure)
    print(f"loss log + curve -> {pred_dir}/loss.txt , {pred_dir}/loss.png")


# ═══════════════════════════ inference ═══════════════════════════
def fit_normalizer(npz_dict, n_train):
    """Build the (normalizer, reference) from a training npz — fit on the first
    n_train cases. Rebuilt every run (checkpoints store only weights + config), so
    the --train-data passed here must match what the model was trained on."""
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


def resolve_viz(n_viz, selected, n_cases):
    """Positions (into the predicted arrays) to render --snapshot / --vtu-out for.
    None -> the first N_VIZ selected cases; 'all' -> every selected case; otherwise
    comma-separated GLOBAL case indices (matching --cases) -> exactly those cases."""
    if n_viz is None:
        return np.arange(min(N_VIZ, n_cases))
    if n_viz == "all":
        return np.arange(n_cases)
    try:
        wanted = [int(part) for part in str(n_viz).split(",")]
    except ValueError:
        raise SystemExit(f"--n-viz must be 'all' or comma-separated case indices, got {n_viz!r}")
    positions = []
    for case in wanted:
        hits = np.where(selected == case)[0]
        if len(hits) == 0:
            raise SystemExit(f"--n-viz {case}: not in the selected set "
                             f"({int(selected.min())}..{int(selected.max())}); adjust --cases")
        positions.append(int(hits[0]))
    return np.array(positions, dtype=int)


def load_model(model_path, device):
    """One load path for any SIREN checkpoint: read weights (+ config if present),
    remap legacy keys, build + load. Returns (model, config).

    - New GeoDONetSIREN checkpoints carry a `config` block (incl. omega_0) — used directly.
    - Legacy original-opnn SIREN checkpoints (`_branch_g`/`_trunk`/`bias`, no config)
      are remapped and the architecture inferred from shapes, with omega_0 = LEGACY_OMEGA_0.
    """
    raw = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    saved_config = raw.get("config") if isinstance(raw, dict) else None
    if is_legacy_state_dict(state_dict):
        state_dict = remap_legacy_state_dict(state_dict)
        config = config_from_state_dict(state_dict, omega_0=LEGACY_OMEGA_0)
        print(f"loaded LEGACY opnn SIREN checkpoint (omega_0 assumed {LEGACY_OMEGA_0:g})")
    elif saved_config is not None:
        config = dict(saved_config)
    else:
        config = config_from_state_dict(state_dict)
        print(f"warning: checkpoint has no config block; assuming omega_0={config['omega_0']:g}")
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

    # tee console output into a log; written to out_dir/test_log.txt when evaluating
    log_lines = []
    def emit(message=""):
        print(message)
        log_lines.append(message)

    model, config = load_model(args.model_path, device)
    emit(f"model: GeoDONetSIREN {config} | {device}")

    # input cases to predict on (theta + Cobiveco required; V_m / time optional)
    data_path = resolve_data_path(args.data_file)
    inputs = subsample_frames(load_inputs(data_path), args.frame_step)

    # normalizer + reference rebuilt from a training npz every run. Source: explicit
    # --train-data, else the eval npz for --test-model or DATA_FILE for --infer.
    if args.train_data is not None:
        train_path = resolve_data_path(args.train_data)
    else:
        train_path = data_path if test_mode else resolve_data_path(None)
    train_npz = (inputs if train_path == data_path and inputs["vm"] is not None
                 else subsample_frames(load_inputs(train_path), args.frame_step))
    normalizer, reference = fit_normalizer(train_npz, args.n_train)
    reference_time = np.asarray(reference["time"], dtype=np.float32)
    emit(f"normalizer rebuilt from {os.path.basename(train_path)} (first {args.n_train} cases)")

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
    emit(f"{'test-model' if test_mode else 'infer'}: {n_cases} cases ({cases_name}), "
         f"{n_nodes} nodes, {len(time_ms)} frames | "
         f"GT {'present' if truth is not None else 'absent'} | {device}")
    if do_eval and truth is None:
        raise SystemExit("evaluation needs ground-truth V_m in --data-path "
                         "(--test-model always evaluates; for --infer, drop --eval)")

    # ── predict per case (bounded memory), timed; denormalize to physical mV ──
    to_device = lambda array: torch.tensor(array, dtype=torch.float32, device=device)
    trunk_chunks = build_trunk_chunks(to_device(normalizer.coords(coords)),
                                      to_device(normalizer.time(time_ms)), CHUNK_FRAMES)
    theta_norm = to_device(normalizer.theta(theta))
    is_cuda = str(device).startswith("cuda")
    preds, infer_times = [], []
    with torch.no_grad():
        if is_cuda:                       # warm up — the first call carries fixed CUDA overhead
            chunked_forward(model, theta_norm[:1], trunk_chunks, n_nodes)
            torch.cuda.synchronize()
        for i in range(n_cases):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = timer.time()
            out = chunked_forward(model, theta_norm[i:i + 1], trunk_chunks, n_nodes)
            if is_cuda:
                torch.cuda.synchronize()
            infer_times.append(timer.time() - t0)
            preds.append(to_numpy(out[0]))
    pred = normalizer.vm_inverse(np.stack(preds)).astype(np.float32)   # (n_cases, n_nodes, n_frames)
    infer_times = np.asarray(infer_times, dtype=np.float64)
    emit(f"inference time: {infer_times.mean() * 1000:.1f} ± {infer_times.std() * 1000:.1f} ms/case "
         f"(total {infer_times.sum():.2f} s for {n_cases} cases)")

    out_dir = args.out_dir or os.path.join("Predictions", checkpoint_stem(args.model_path))

    # ── evaluation (metrics always print; arrays go into predictions.npz if --save) ──
    at_pred = at_true = None
    metrics = {}
    if do_eval:
        vm_rel, vm_mae = vm_rel_l2_mae(pred, truth)
        at_pred = np.stack([activation_time(pred[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
        at_true = np.stack([activation_time(truth[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
        at_rel, at_mae = at_rel_l2_mae(at_pred, at_true)
        emit(format_metric_table("V_m", case_names, vm_rel, vm_mae, "Rel L2", "MAE (mV)"))
        emit(format_metric_table("AT", case_names, at_rel, at_mae, "Rel L2", "MAE (ms)"))
        emit(format_metric_summary(n_cases, cases_name, vm_rel, vm_mae, at_rel, at_mae))
        metrics = dict(vm_rel_l2=vm_rel, vm_mae=vm_mae, at_pred=at_pred, at_true=at_true,
                       at_rel_l2=at_rel, at_mae=at_mae)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "test_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(log_lines) + "\n")
        print(f"  test log -> {log_path}")

    # cases to render for --snapshot / --vtu-out (first --n-viz, or chosen case indices)
    viz_idx = (resolve_viz(args.n_viz, selected, n_cases)
               if (args.snapshot or args.vtu_out) else np.array([], dtype=int))

    # ── --snapshot: V_m(t) traces (xyz-free) + 3D V_m/AT scatter SVGs (needs --reference-xyz) ──
    if args.snapshot:
        traces_root = os.path.join(out_dir, "traces")
        for c in viz_idx:
            case_dir = os.path.join(traces_root, labels[c]); os.makedirs(case_dir, exist_ok=True)
            save_vm_traces(pred[c], None if truth is None else truth[c], time_ms,
                           os.path.join(case_dir, "traces.png"))
        print(f"  traces ({len(viz_idx)} cases) -> {traces_root}/<case>")
        if os.path.exists(args.reference_xyz):
            xyz = np.load(args.reference_xyz)["cartesian_coords"].astype(np.float32)
            if xyz.shape[0] != n_nodes:
                raise SystemExit(f"reference xyz has {xyz.shape[0]} nodes != {n_nodes} predicted")
            if at_pred is None:        # AT not computed in the eval block (e.g. --infer w/o --eval)
                at_pred = np.stack([activation_time(pred[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
                if truth is not None:
                    at_true = np.stack([activation_time(truth[c], time_ms, AT_THRESHOLD_MV) for c in range(n_cases)])
            render_snapshots(out_dir, xyz, pred[viz_idx],
                             truth[viz_idx] if truth is not None else None,
                             at_pred[viz_idx],
                             at_true[viz_idx] if at_true is not None else None,
                             time_ms, [labels[c] for c in viz_idx])
        else:
            print(f"  3D scatter skipped (--reference-xyz not found: {args.reference_xyz})")

    # ── --vtu-out: predicted V_m .vtu+.pvd series per rendered case (+ GT/abserr when GT present) ──
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
        for c in viz_idx:
            fields = {"Vm": pred[c]}
            if truth is not None:
                fields["Vm_gt"] = truth[c]
                fields["Vm_abserr"] = np.abs(truth[c] - pred[c]).astype(np.float32)
            write_vtu_series(os.path.join(vtu_root, labels[c]), points, tetra, time_ms, fields)
        print(f"  wrote {len(viz_idx)} V_m .vtu series ({len(time_ms)} frames each) -> {vtu_root}")

    # ── --color-bar ──
    if args.color_bar:
        os.makedirs(out_dir, exist_ok=True)
        save_colorbars(out_dir, at_pred, at_true)

    # ── --save: full predictions.npz ──
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
        saved = {"pred": pred, "time": time_ms, "case_names": np.asarray(labels),
                 "infer_time_per_case_s": infer_times}
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
