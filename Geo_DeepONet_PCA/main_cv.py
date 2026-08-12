"""Leakage-safe five-fold CV for waveform-supervised Geo-DeepONet-PCA.

Each fold uses 95 fit, 5 validation, and 25 test hearts.  The phase-aligned
node template, temporal PCA modes, normalization statistics, and V_m scale are
fit from the 95 fitting hearts only.  Test hearts never enter a fold decoder.

To avoid calculating a 601x601 temporal covariance five times, all waveforms
are aligned once to a fixed, fold-independent reference time. Per-heart
uncentred scatter matrices are then combined algebraically to obtain each
fold's exact node-centred training covariance.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import physical_features, sampled_vm_target, set_seed, tensor
from opnn import FeatureDeepONet
from utils import (DifferentiablePhaseDecoder, activation_time, at_metrics,
                   decode_features, median_max_dvdt, shift_waveforms,
                   vm_metrics)


FEATURE_DATA = "/home/svu/e1032484/scratch/geo_deeponet_pca_f601_k5.npz"
VM_DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
N_FOLDS, N_VAL = 5, 5
AT_THRESHOLD = -10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-fold CV for decoded-V_m-supervised Geo-DeepONet-PCA")
    parser.add_argument("--data", default=FEATURE_DATA,
                        help="compact archive; AT is reused but PCA targets are refit per fold")
    parser.add_argument("--vm-data", default=VM_DATA)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nodes-per-step", type=int, default=2_048)
    parser.add_argument("--val-nodes", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-nodes", type=int, default=20_000)
    parser.add_argument("--reference-at", type=float, default=80.0,
                        help="fixed fold-independent alignment reference in ms")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def validate_args(args, n_cases=None, n_nodes=None):
    positive = {
        "--folds": args.folds, "--n-val": args.n_val,
        "--epochs": args.epochs, "--batch-size": args.batch_size,
        "--nodes-per-step": args.nodes_per_step,
        "--val-nodes": args.val_nodes, "--width": args.width,
        "--depth": args.depth, "--n-components": args.n_components,
        "--val-every": args.val_every, "--log-every": args.log_every,
        "--chunk-nodes": args.chunk_nodes,
    }
    bad = [name for name, value in positive.items() if value < 1]
    if bad:
        raise SystemExit(f"arguments must be positive: {', '.join(bad)}")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if n_cases is not None:
        if n_cases % args.folds:
            raise SystemExit(f"{n_cases} hearts cannot divide evenly into {args.folds} folds")
        remaining = n_cases - n_cases // args.folds
        if args.n_val >= remaining:
            raise SystemExit("validation split leaves no fitting hearts")
    if n_nodes is not None and max(args.nodes_per_step, args.val_nodes) > n_nodes:
        raise SystemExit(f"node sample cannot exceed {n_nodes}")


def make_folds(n_cases, n_folds, n_val, seed):
    """Match the shuffled protocol used by Geo_DONet and Geo_MLP CV."""
    indices = np.arange(n_cases)
    random_state = np.random.RandomState(seed)
    random_state.shuffle(indices)
    fold_size = n_cases // n_folds
    folds = []
    for fold in range(n_folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        test_idx = indices[start:end]
        remaining = np.concatenate((indices[:start], indices[end:]))
        val_idx = remaining[-n_val:]
        train_idx = remaining[:-n_val]
        folds.append((train_idx, val_idx, test_idx))
    return indices, folds


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan
    return float(values[finite].mean()), float(values[finite].std())


def stat(values, digits=4):
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def build_shared_alignment(vm, at_all, dt, reference_at, chunk_nodes):
    """Align once and calculate each heart's uncentred temporal scatter."""
    n_cases, n_nodes, n_frames = vm.shape
    aligned = np.empty_like(vm, dtype=np.float32)
    start_time = time.time()
    print(f"aligning {n_cases} hearts once at fixed reference {reference_at:g} ms ...",
          flush=True)
    for case in range(n_cases):
        if not np.isfinite(at_all[case]).all():
            bad = np.count_nonzero(~np.isfinite(at_all[case]))
            raise SystemExit(f"case {case}: {bad} nodes have invalid activation time")
        shifts = at_all[case] - np.float32(reference_at)
        for start in range(0, n_nodes, chunk_nodes):
            end = min(start + chunk_nodes, n_nodes)
            aligned[case, start:end] = shift_waveforms(
                vm[case, start:end], shifts[start:end], dt)
        if (case + 1) % 10 == 0 or case + 1 == n_cases:
            elapsed = time.time() - start_time
            eta = elapsed / (case + 1) * (n_cases - case - 1)
            print(f"  alignment {case + 1:3d}/{n_cases} | "
                  f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)

    print("building shared temporal sufficient statistics ...", flush=True)
    # A fixed zero origin keeps this shared calculation completely independent
    # of every fold's labels. Float64 provides ample precision for the later
    # subtraction of the fold-specific node-template contribution.
    origin = np.zeros((n_nodes, n_frames), dtype=np.float32)
    case_scatter = np.zeros((n_cases, n_frames, n_frames), dtype=np.float64)
    scatter_start = time.time()
    for case in range(n_cases):
        gram = case_scatter[case]
        for start in range(0, n_nodes, chunk_nodes):
            end = min(start + chunk_nodes, n_nodes)
            block = (aligned[case, start:end] - origin[start:end]).astype(np.float64)
            gram += block.T @ block
        if (case + 1) % 5 == 0 or case + 1 == n_cases:
            elapsed = time.time() - scatter_start
            eta = elapsed / (case + 1) * (n_cases - case - 1)
            print(f"  scatter   {case + 1:3d}/{n_cases} | "
                  f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)
    return aligned, origin, case_scatter


def fit_fold_basis(aligned, origin, case_scatter, train_idx, time_ms,
                   reference_at, n_components, chunk_nodes):
    """Fit a node template and temporal PCA using only ``train_idx``."""
    n_train = len(train_idx)
    n_nodes, n_frames = aligned.shape[1:]
    node_sum = np.zeros((n_nodes, n_frames), dtype=np.float64)
    scatter = np.zeros((n_frames, n_frames), dtype=np.float64)
    for case in train_idx:
        node_sum += aligned[case]
        scatter += case_scatter[case]
    node_template = (node_sum / n_train).astype(np.float32)
    del node_sum

    # Convert scatter around the shared origin to scatter around this fold's
    # node template. This is algebraically identical to a new residual pass.
    for start in range(0, n_nodes, chunk_nodes):
        end = min(start + chunk_nodes, n_nodes)
        delta = (node_template[start:end] - origin[start:end]).astype(np.float64)
        scatter -= n_train * (delta.T @ delta)
    scatter = 0.5 * (scatter + scatter.T)
    n_samples = n_train * n_nodes
    covariance = scatter / max(n_samples - 1, 1)
    eigval, eigvec = np.linalg.eigh(covariance)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0.0, None)
    components = eigvec[:, order[:n_components]].astype(np.float32)
    total = eigval.sum()
    evr = eigval[:n_components] / total if total > 0 else np.zeros(n_components)
    pca_std = np.sqrt(eigval[:n_components] * (n_samples - 1) / n_samples)
    pca_std = pca_std.astype(np.float32)
    pca_std[pca_std < 1e-8] = 1.0
    basis = dict(node_template=node_template,
                 residual_mean=np.zeros(n_frames, dtype=np.float32),
                 components=components, time=time_ms.astype(np.float32),
                 reference_at=float(reference_at), at_threshold=AT_THRESHOLD)
    return basis, eigval[:n_components], evr, pca_std


def fold_vm_scale(vm, train_idx):
    """Training-only population standard deviation without a multi-GB copy."""
    count = 0
    total = 0.0
    total_square = 0.0
    for case in train_idx:
        values = vm[case]
        count += values.size
        total += float(np.sum(values, dtype=np.float64))
        total_square += float(np.sum(np.square(values, dtype=np.float64),
                                     dtype=np.float64))
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return float(np.sqrt(variance))


def save_loss_plot(rows, path, fold):
    values = np.asarray(rows, dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].semilogy(values[:, 0], values[:, 1], label="train V_m")
    valid = np.isfinite(values[:, 2])
    axes[0].semilogy(values[valid, 0], values[valid, 2], label="validation V_m")
    axes[0].set_ylabel("normalized V_m MSE")
    axes[1].semilogy(values[:, 0], values[:, 3], label="train AT diagnostic")
    axes[1].semilogy(values[valid, 0], values[valid, 4], label="validation AT diagnostic")
    axes[1].set_ylabel("normalized AT MSE (diagnostic only)")
    for axis in axes:
        axis.set_xlabel("epoch"); axis.grid(alpha=0.3); axis.legend()
    figure.suptitle(f"Geo_DeepONet_PCA CV fold {fold}")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def save_basis(path, basis, eigval, evr, train_idx):
    np.savez(path, node_template=basis["node_template"],
             residual_mean=basis["residual_mean"],
             components=basis["components"], time=basis["time"],
             reference_at=np.float32(basis["reference_at"]),
             at_threshold=np.float32(basis["at_threshold"]),
             eigval=eigval, evr=evr, train_idx=np.asarray(train_idx),
             pca_center=np.asarray("node"),
             shift_interpolation=np.asarray("linear_constant_edge"))


def train_fold(fold, train_idx, val_idx, theta, coords, at_all, vm, basis,
               pca_std, args, fold_dir, device):
    set_seed(args.seed + fold)
    theta_mean = theta[train_idx].mean(axis=0).astype(np.float32)
    theta_std = theta[train_idx].std(axis=0).astype(np.float32)
    theta_std[theta_std < 1e-8] = 1.0
    coord_min = coords.min(axis=0).astype(np.float32)
    coord_max = coords.max(axis=0).astype(np.float32)
    coords_norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)

    at_train = at_all[train_idx]
    target_mean = np.concatenate(([at_train.mean()],
                                  np.zeros(args.n_components))).astype(np.float32)
    target_std = np.concatenate(([at_train.std()], pca_std)).astype(np.float32)
    target_std[target_std < 1e-8] = 1.0
    vm_scale = fold_vm_scale(vm, train_idx)

    theta_train = tensor((theta[train_idx] - theta_mean) / theta_std, device)
    theta_val = tensor((theta[val_idx] - theta_mean) / theta_std, device)
    target_mean_t = tensor(target_mean, device).view(1, 1, -1)
    target_std_t = tensor(target_std, device).view(1, 1, -1)
    decoder = DifferentiablePhaseDecoder(basis, args.n_components).to(device)
    model = FeatureDeepONet(theta.shape[1], coords.shape[1], args.width,
                            args.depth, args.n_components + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    val_rng = np.random.default_rng(args.seed + 10_000 + fold)
    val_nodes = np.sort(val_rng.choice(len(coords), args.val_nodes, replace=False))
    val_coords = tensor(coords_norm[val_nodes], device)
    val_truth = sampled_vm_target(vm, val_idx, val_nodes, device)
    val_at_norm = tensor((at_all[val_idx][:, val_nodes] - target_mean[0]) /
                         target_std[0], device)

    checkpoint_path = os.path.join(fold_dir, "model.pt")
    loss_path = os.path.join(fold_dir, "loss.csv")
    best_val, best_epoch = float("inf"), -1
    order = np.arange(len(train_idx))
    rows = []
    fold_start = time.time()
    print(f"  model: w{args.width} d{args.depth}, {n_params:,} params; "
          f"V_m scale {vm_scale:.5g} mV", flush=True)
    print(f"  target std: AT={target_std[0]:.4g}, "
          + ", ".join(f"PC{k + 1}={target_std[k + 1]:.4g}"
                      for k in range(args.n_components)), flush=True)

    with open(loss_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_vm_mse", "val_vm_mse",
                         "train_at_mse_diagnostic", "val_at_mse_diagnostic"))
        for epoch in range(args.epochs):
            model.train(); np.random.shuffle(order)
            running_vm = running_at = 0.0
            batches = 0
            for start in range(0, len(order), args.batch_size):
                local_batch = order[start:start + args.batch_size]
                global_batch = train_idx[local_batch]
                nodes = np.random.choice(len(coords), args.nodes_per_step,
                                         replace=False)
                prediction_norm = model(theta_train[local_batch],
                                        tensor(coords_norm[nodes], device))
                prediction = physical_features(prediction_norm,
                                               target_mean_t, target_std_t)
                decoded = decoder(prediction, nodes)
                truth = sampled_vm_target(vm, global_batch, nodes, device)
                vm_loss = torch.mean(((decoded - truth) / vm_scale) ** 2)
                at_target = tensor((at_all[global_batch][:, nodes] - target_mean[0]) /
                                   target_std[0], device)
                at_diag = torch.mean((prediction_norm[..., 0] - at_target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                vm_loss.backward(); optimizer.step()
                running_vm += float(vm_loss.item())
                running_at += float(at_diag.item())
                batches += 1
            train_vm = running_vm / batches
            train_at = running_at / batches

            val_vm = val_at = np.nan
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_prediction_norm = model(theta_val, val_coords)
                    val_prediction = physical_features(val_prediction_norm,
                                                       target_mean_t, target_std_t)
                    val_decoded = decoder(val_prediction, val_nodes)
                    val_vm = float(torch.mean(((val_decoded - val_truth) /
                                               vm_scale) ** 2).item())
                    val_at = float(torch.mean((val_prediction_norm[..., 0] -
                                               val_at_norm) ** 2).item())
                if val_vm < best_val:
                    best_val, best_epoch = val_vm, epoch
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": model.config(), "fold": fold,
                        "train_idx": train_idx, "val_idx": val_idx,
                        "theta_mean": theta_mean, "theta_std": theta_std,
                        "coord_min": coord_min, "coord_max": coord_max,
                        "target_mean": target_mean, "target_std": target_std,
                        "vm_scale": vm_scale,
                        "basis_file": os.path.abspath(os.path.join(fold_dir, "basis.npz")),
                        "best_val": best_val, "best_epoch": best_epoch,
                    }, checkpoint_path)
            rows.append((epoch, train_vm, val_vm, train_at, val_at))
            writer.writerow(rows[-1])
            if epoch % args.log_every == 0 or epoch == args.epochs - 1:
                handle.flush()
                elapsed = time.time() - fold_start
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(f"  fold {fold} epoch {epoch:4d}/{args.epochs} | "
                      f"train Vm {train_vm:.6f} ATdiag {train_at:.5f} | "
                      f"val Vm {val_vm:.6f} ATdiag {val_at:.5f} | "
                      f"best {best_val:.6f}@{best_epoch} | eta {eta / 60:.1f} min",
                      flush=True)

    save_loss_plot(rows, os.path.join(fold_dir, "loss.png"), fold)
    return checkpoint_path, best_val, best_epoch, (time.time() - fold_start) / 60


def oracle_pca_mae(predicted_coefficients, aligned_truth, basis, chunk_nodes):
    total = np.zeros(predicted_coefficients.shape[1], dtype=np.float64)
    n_nodes = len(predicted_coefficients)
    for start in range(0, n_nodes, chunk_nodes):
        end = min(start + chunk_nodes, n_nodes)
        residual = (aligned_truth[start:end] - basis["node_template"][start:end]
                    - basis["residual_mean"])
        oracle = residual @ basis["components"]
        total += np.abs(predicted_coefficients[start:end] - oracle).sum(axis=0)
    return total / n_nodes


def evaluate_fold(fold, test_idx, theta, coords, at_all, vm, aligned, time_ms,
                  case_names, basis, checkpoint_path, args, fold_dir, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FeatureDeepONet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    coords_norm = ((coords - checkpoint["coord_min"]) /
                   (checkpoint["coord_max"] - checkpoint["coord_min"] + 1e-8))
    coords_t = tensor(coords_norm, device)
    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]
    dt = float(np.median(np.diff(time_ms)))
    metric_keys = ("vm_l2", "vm_mae", "direct_at_l2", "direct_at_mae",
                   "decoded_at_mae", "dvdt_fraction", "inference_time_s")
    metrics = {key: [] for key in metric_keys}
    pca_mae = []
    rows = []
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    with torch.no_grad():
        for position, case in enumerate(test_idx):
            theta_norm = ((theta[case:case + 1] - checkpoint["theta_mean"]) /
                          checkpoint["theta_std"])
            sync(); infer_start = time.perf_counter()
            prediction_norm = model(tensor(theta_norm, device), coords_t)
            sync(); inference_time = time.perf_counter() - infer_start
            prediction = (prediction_norm[0].cpu().numpy() * target_std
                          + target_mean).astype(np.float32)
            decoded = decode_features(prediction, basis, args.chunk_nodes)
            truth = vm[case]
            vm_l2, vm_mae = vm_metrics(decoded, truth)
            direct_l2, direct_mae = at_metrics(prediction[:, 0], at_all[case])
            decoded_at = activation_time(decoded, time_ms, AT_THRESHOLD)
            _, decoded_mae = at_metrics(decoded_at, at_all[case])
            slope_fraction = (median_max_dvdt(decoded, dt) /
                              median_max_dvdt(truth, dt))
            pc_mae = oracle_pca_mae(prediction[:, 1:], aligned[case], basis,
                                     args.chunk_nodes)
            values = (vm_l2, vm_mae, direct_l2, direct_mae, decoded_mae,
                      slope_fraction, inference_time)
            for key, value in zip(metric_keys, values):
                metrics[key].append(value)
            pca_mae.append(pc_mae)
            rows.append((fold, int(case), str(case_names[case]), *values, *pc_mae))
            print(f"  fold {fold} test {position + 1:2d}/{len(test_idx)} "
                  f"{case_names[case]} | Vm MAE {vm_mae:.3f} mV | "
                  f"AT {direct_mae:.3f}/{decoded_mae:.3f} ms | "
                  f"dVdt {slope_fraction:.3f}", flush=True)
            del decoded

    for key in metrics:
        metrics[key] = np.asarray(metrics[key], dtype=np.float32)
    metrics["pca_mae"] = np.asarray(pca_mae, dtype=np.float32)
    metrics["test_idx"] = np.asarray(test_idx)
    with open(os.path.join(fold_dir, "test_metrics.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "global_case", "case_name", *metric_keys,
                         *(f"pca_{k + 1}_mae" for k in range(args.n_components))))
        writer.writerows(rows)
    del model, coords_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def save_summary_plot(fold_results, path):
    specs = (("vm_l2", "V_m relative L2"), ("vm_mae", "V_m MAE (mV)"),
             ("direct_at_mae", "direct AT MAE (ms)"),
             ("decoded_at_mae", "decoded AT MAE (ms)"),
             ("dvdt_fraction", "max-dV/dt fraction"))
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    x = np.arange(len(fold_results))
    for axis, (key, label) in zip(axes.ravel(), specs):
        values = [mean_std(result[key]) for result in fold_results]
        axis.errorbar(x, [v[0] for v in values], yerr=[v[1] for v in values],
                      fmt="o", capsize=4)
        pooled = np.concatenate([result[key] for result in fold_results])
        axis.axhline(np.nanmean(pooled), color="C3", ls="--", label="pooled mean")
        axis.set_xticks(x); axis.set_xlabel("test fold"); axis.set_ylabel(label)
        axis.grid(alpha=0.3); axis.legend()
    axes.ravel()[-1].axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def main():
    args = parse_args(); validate_args(args); set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    compact = np.load(args.data, allow_pickle=True)
    theta = compact["theta"].astype(np.float32)
    coords = compact["coords"].astype(np.float32)
    at_all = compact["targets"][..., 0].astype(np.float32)
    case_names = (compact["case_names"].astype(str) if "case_names" in compact.files
                  else np.asarray([f"case{i}" for i in range(len(theta))]))
    compact_time = compact["time"].astype(np.float32)

    print(f"loading raw V_m: {args.vm_data}", flush=True)
    vm_archive = np.load(args.vm_data, allow_pickle=True)
    vm = vm_archive["vm"]
    time_ms = vm_archive["time"].astype(np.float32)
    if vm.shape[:2] != (len(theta), len(coords)):
        raise SystemExit("compact feature and raw V_m shapes disagree")
    if not np.array_equal(time_ms, compact_time):
        raise SystemExit("compact feature and raw V_m time grids disagree")
    dt = float(np.median(np.diff(time_ms)))
    validate_args(args, len(theta), len(coords))
    shuffled, folds = make_folds(len(theta), args.folds, args.n_val, args.seed)

    default_dir = (f"CV_{args.folds}fold_{args.epochs}ep_w{args.width}_d{args.depth}_"
                   f"n{args.nodes_per_step}_f{len(time_ms)}_vmloss")
    out_dir = args.out_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"output: {out_dir}")
    print(f"data: {len(theta)} hearts, {len(coords)} nodes, {len(time_ms)} frames")
    print(f"protocol: {args.folds} folds; 95 fit / {args.n_val} val / "
          f"{len(theta) // args.folds} test; no early stopping")
    print(f"training: {args.epochs} epochs/fold, batch {args.batch_size}, "
          f"nodes {args.nodes_per_step}, lr {args.lr:g} | {device}")

    total_start = time.time()
    aligned, origin, case_scatter = build_shared_alignment(
        vm, at_all, dt, args.reference_at, args.chunk_nodes)

    fold_results = []
    fold_meta = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n=== fold {fold}: {len(train_idx)} fit / {len(val_idx)} val / "
              f"{len(test_idx)} test ===", flush=True)
        basis, eigval, evr, pca_std = fit_fold_basis(
            aligned, origin, case_scatter, train_idx, time_ms,
            args.reference_at, args.n_components, args.chunk_nodes)
        basis_path = os.path.join(fold_dir, "basis.npz")
        save_basis(basis_path, basis, eigval, evr, train_idx)
        print("  PCA EVR: " + ", ".join(f"PC{k + 1}={value:.4f}"
                                        for k, value in enumerate(evr)), flush=True)
        checkpoint, best_val, best_epoch, train_minutes = train_fold(
            fold, train_idx, val_idx, theta, coords, at_all, vm, basis,
            pca_std, args, fold_dir, device)
        result = evaluate_fold(fold, test_idx, theta, coords, at_all, vm,
                               aligned, time_ms, case_names, basis, checkpoint,
                               args, fold_dir, device)
        fold_results.append(result)
        fold_meta.append((best_val, best_epoch, train_minutes))
        print(f"  fold {fold} result: Vm {stat(result['vm_mae'], 3)} mV | "
              f"direct AT {stat(result['direct_at_mae'], 3)} ms | "
              f"decoded AT {stat(result['decoded_at_mae'], 3)} ms | "
              f"dVdt {stat(result['dvdt_fraction'], 3)}", flush=True)
        del basis

    scalar_keys = ("vm_l2", "vm_mae", "direct_at_l2", "direct_at_mae",
                   "decoded_at_mae", "dvdt_fraction", "inference_time_s")
    pooled = {key: np.concatenate([result[key] for result in fold_results])
              for key in scalar_keys}
    pooled_pca = np.concatenate([result["pca_mae"] for result in fold_results])
    all_test_idx = np.concatenate([result["test_idx"] for result in fold_results])
    fold_id = np.concatenate([np.full(len(result["test_idx"]), fold, dtype=np.int32)
                              for fold, result in enumerate(fold_results)])
    total_minutes = (time.time() - total_start) / 60

    lines = ["=" * 72,
             f"Geo_DeepONet_PCA {args.folds}-fold pooled (N={len(all_test_idx)} hearts)",
             "Fold-specific train-only decoder basis and normalization",
             f"V_m Rel L2       : {stat(pooled['vm_l2'])}",
             f"V_m MAE          : {stat(pooled['vm_mae'], 3)} mV",
             f"direct AT Rel L2 : {stat(pooled['direct_at_l2'])}",
             f"direct AT MAE    : {stat(pooled['direct_at_mae'], 3)} ms",
             f"decoded AT MAE   : {stat(pooled['decoded_at_mae'], 3)} ms",
             f"upstroke fraction: {stat(pooled['dvdt_fraction'], 3)}",
             f"feature inference: {stat(pooled['inference_time_s'], 4)} s/heart",
             "PCA feature MAE   : " + ", ".join(
                 f"PC{k + 1}={stat(pooled_pca[:, k], 3)}"
                 for k in range(args.n_components)), "", "Per fold:"]
    for fold, result in enumerate(fold_results):
        best_val, best_epoch, minutes = fold_meta[fold]
        lines.append(f"  fold {fold}: Vm MAE={stat(result['vm_mae'], 3)} mV, "
                     f"direct AT={stat(result['direct_at_mae'], 3)} ms, "
                     f"dVdt={stat(result['dvdt_fraction'], 3)}, "
                     f"best={best_val:.6f}@{best_epoch}, train={minutes:.1f} min")
    lines.append(f"\nTotal wall time: {total_minutes:.1f} min")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")

    with open(os.path.join(out_dir, "per_case_metrics.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "global_case", "case_name", *scalar_keys,
                         *(f"pca_{k + 1}_mae" for k in range(args.n_components))))
        for fold, result in enumerate(fold_results):
            for local, case in enumerate(result["test_idx"]):
                writer.writerow((fold, int(case), case_names[case],
                                 *(result[key][local] for key in scalar_keys),
                                 *result["pca_mae"][local]))

    np.savez_compressed(os.path.join(out_dir, "cv_results.npz"),
                        test_idx=all_test_idx, fold_id=fold_id,
                        case_names=case_names[all_test_idx], fold_indices=shuffled,
                        pca_mae=pooled_pca,
                        best_val=np.asarray([v[0] for v in fold_meta]),
                        best_epoch=np.asarray([v[1] for v in fold_meta]),
                        train_minutes=np.asarray([v[2] for v in fold_meta]),
                        **pooled)
    save_summary_plot(fold_results, os.path.join(out_dir, "cv_summary.png"))
    print(f"saved pooled outputs -> {out_dir}")


if __name__ == "__main__":
    main()
