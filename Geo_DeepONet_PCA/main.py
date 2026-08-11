"""Train/evaluate Geo-DeepONet for AT + aligned-PCA waveform features."""
import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from opnn import FeatureDeepONet, initialise_from_at_checkpoint
from utils import (FeatureNormalizer, activation_time, at_metrics, decode_features,
                   load_decoder_basis, load_feature_data, median_max_dvdt,
                   split_indices, to_numpy, vm_metrics)


DATA = "/home/svu/e1032484/scratch/geo_deeponet_pca_f601_k5.npz"
BASIS = "/home/svu/e1032484/scratch/pca_phase_aligned_basis_f601.npz"
VM_DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
N_TRAIN, N_VAL = 95, 5


def parse_args():
    parser = argparse.ArgumentParser(description="Geo-DeepONet: AT + aligned-PCA features")
    parser.add_argument("--test-model", action="store_true")
    parser.add_argument("--data", default=DATA, help="compact output from prepare_data.py")
    parser.add_argument("--basis", default=BASIS, help="phase-aligned decoder basis")
    parser.add_argument("--vm-data", default=VM_DATA, help="raw V_m archive for test decoding")
    parser.add_argument("--skip-vm-eval", action="store_true",
                        help="evaluate predicted AT/PCA features only; do not load 11 GB V_m npz")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--init-at-checkpoint", default=None,
                        help="optional legacy Geo_DeepONet AT checkpoint for encoder transfer")
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--at-loss-weight", type=float, default=0.5,
                        help="relative weight of the AT loss group (normalised with PCA weight)")
    parser.add_argument("--pca-loss-weight", type=float, default=0.5,
                        help="relative weight of the combined PCA loss group")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5000,
                        help="early-stop epochs without val improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="default: cuda when available")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-viz", type=int, default=2)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--chunk-nodes", type=int, default=20_000,
                        help="decoder interpolation chunk size")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def default_stem(args):
    return f"geodeeponet_pca_k{args.n_components}_w{args.width}_d{args.depth}_{args.epochs}ep"


def save_loss_plot(rows, path):
    values = np.asarray(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    valid = np.isfinite(values[:, 4])
    axes[0].semilogy(values[:, 0], values[:, 1], label="train total")
    axes[0].semilogy(values[valid, 0], values[valid, 4], label="validation total")
    axes[0].set_title("weighted objective")
    axes[1].semilogy(values[:, 0], values[:, 2], label="train AT")
    axes[1].semilogy(values[:, 0], values[:, 3], label="train PCA")
    axes[1].semilogy(values[valid, 0], values[valid, 5], "--", label="validation AT")
    axes[1].semilogy(values[valid, 0], values[valid, 6], "--", label="validation PCA")
    axes[1].set_title("unweighted group MSE")
    for axis in axes:
        axis.set_xlabel("epoch"); axis.set_ylabel("normalised MSE")
        axis.legend(); axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def load_inputs(args):
    data = load_feature_data(args.data)
    needed = args.n_components + 1
    if data["targets"].shape[2] < needed:
        raise SystemExit(f"feature data has {data['targets'].shape[2] - 1} PCA components; "
                         f"requested {args.n_components}. Re-run prepare_data.py.")
    data["targets"] = data["targets"][:, :, :needed]
    data["target_names"] = data["target_names"][:needed]
    return data


def make_loss_weights(args, device):
    """Return normalised AT/PCA group weights and within-PCA EVR weights."""
    if args.at_loss_weight < 0 or args.pca_loss_weight < 0:
        raise SystemExit("loss-group weights must be non-negative")
    group_total = args.at_loss_weight + args.pca_loss_weight
    if group_total <= 0:
        raise SystemExit("at least one loss-group weight must be positive")
    at_group_weight = args.at_loss_weight / group_total
    pca_group_weight = args.pca_loss_weight / group_total

    basis = np.load(args.basis, allow_pickle=False)
    if "evr" not in basis.files or len(basis["evr"]) < args.n_components:
        raise SystemExit(f"basis needs at least {args.n_components} explained-variance values")
    pca_weights = basis["evr"][:args.n_components].astype(np.float64)
    if not np.isfinite(pca_weights).all() or pca_weights.sum() <= 0:
        raise SystemExit("basis PCA explained-variance weights are invalid")
    pca_weights /= pca_weights.sum()
    return (float(at_group_weight), float(pca_group_weight),
            tensor(pca_weights.astype(np.float32), device), pca_weights)


def grouped_feature_loss(prediction, target, at_group_weight,
                         pca_group_weight, pca_weights):
    """AT and PCA are equal-level groups; PCA modes are EVR-weighted within their group."""
    channel_mse = ((prediction - target) ** 2).mean(dim=(0, 1))
    at_mse = channel_mse[0]
    pca_mse = (channel_mse[1:] * pca_weights).sum()
    total = at_group_weight * at_mse + pca_group_weight * pca_mse
    return total, at_mse, pca_mse


def train(args, device):
    data = load_inputs(args)
    theta, coords, targets = data["theta"], data["coords"], data["targets"]
    train_idx, val_idx, _ = split_indices(len(theta), args.n_train, args.n_val)
    normalizer = FeatureNormalizer(theta[train_idx], coords, targets[train_idx])

    theta_train = tensor(normalizer.theta(theta[train_idx]), device)
    target_train = tensor(normalizer.targets(targets[train_idx]), device)
    theta_val = tensor(normalizer.theta(theta[val_idx]), device)
    target_val = tensor(normalizer.targets(targets[val_idx]), device)
    coords_tensor = tensor(normalizer.coords(coords), device)
    at_group_weight, pca_group_weight, pca_weights, pca_weights_np = \
        make_loss_weights(args, device)

    model = FeatureDeepONet(theta.shape[1], coords.shape[1], args.width, args.depth,
                            targets.shape[2]).to(device)
    if args.init_at_checkpoint:
        copied = initialise_from_at_checkpoint(model, args.init_at_checkpoint, device)
        print(f"transferred {copied} branch/trunk tensors from {args.init_at_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stem = default_stem(args)
    model_path = args.model_path or os.path.join("CheckPts", stem + ".pt")
    out_dir = args.out_dir or os.path.join("Predictions", stem)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"data: {len(theta)} hearts, {len(coords)} nodes, targets {list(data['target_names'])}")
    print(f"split: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(theta) - len(train_idx) - len(val_idx)} test")
    print(f"model: {model.config()} | {n_params:,} params | {device}")
    print(f"loss groups: AT {at_group_weight:.3f} / PCA {pca_group_weight:.3f}")
    print("PCA weights within group: "
          + ", ".join(f"PC{k + 1}={weight:.4f}" for k, weight in enumerate(pca_weights_np)))
    print(f"checkpoint -> {model_path}")

    history = []
    best_val = float("inf")
    best_epoch = -1
    order = np.arange(len(train_idx))
    tic = time.time()
    loss_path = os.path.join(out_dir, "loss.csv")
    with open(loss_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_total", "train_at_mse", "train_pca_mse",
                         "val_total", "val_at_mse", "val_pca_mse"])
        for epoch in range(args.epochs):
            model.train(); np.random.shuffle(order)
            running_total = running_at = running_pca = 0.0
            n_batches = 0
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                prediction = model(theta_train[batch], coords_tensor)
                loss, at_loss, pca_loss = grouped_feature_loss(
                    prediction, target_train[batch], at_group_weight,
                    pca_group_weight, pca_weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                running_total += loss.item()
                running_at += at_loss.item()
                running_pca += pca_loss.item()
                n_batches += 1
            train_total = running_total / n_batches
            train_at = running_at / n_batches
            train_pca = running_pca / n_batches

            val_total = val_at = val_pca = np.nan
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_prediction = model(theta_val, coords_tensor)
                    val_loss = grouped_feature_loss(val_prediction, target_val,
                                                    at_group_weight, pca_group_weight,
                                                    pca_weights)
                    val_total, val_at, val_pca = (value.item() for value in val_loss)
                if val_total < best_val:
                    best_val, best_epoch = val_total, epoch
                    torch.save(dict(model_state_dict=model.state_dict(), config=model.config(),
                                    normalizer=normalizer.state(),
                                    target_names=data["target_names"],
                                    loss_config=dict(at_group_weight=at_group_weight,
                                                     pca_group_weight=pca_group_weight,
                                                     pca_weights=pca_weights_np),
                                    n_train=args.n_train, n_val=args.n_val), model_path)
            history.append((epoch, train_total, train_at, train_pca,
                            val_total, val_at, val_pca))
            writer.writerow(history[-1]); handle.flush()

            if epoch % args.print_every == 0 or epoch == args.epochs - 1:
                elapsed = time.time() - tic
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                message = (f"epoch {epoch:5d}/{args.epochs} | "
                           f"train {train_total:.6f} (AT {train_at:.6f}, PCA {train_pca:.6f}) | "
                           f"val {val_total:.6f} (AT {val_at:.6f}, PCA {val_pca:.6f}) | "
                           f"best {best_val:.6f} @ {best_epoch} | "
                           f"eta {eta / 60:.1f} min")
                if np.isfinite(val_total):
                    physical = normalizer.targets_inverse(to_numpy(val_prediction))
                    at_mae = np.abs(physical[..., 0] - targets[val_idx, :, 0]).mean()
                    message += f" | val AT MAE {at_mae:.3f} ms"
                print(message, flush=True)

            if args.patience > 0 and best_epoch >= 0 and epoch - best_epoch >= args.patience:
                print(f"early stop: no validation improvement for {args.patience} epochs")
                break

    save_loss_plot(history, os.path.join(out_dir, "loss.png"))
    print(f"done in {(time.time() - tic) / 60:.1f} min | best val {best_val:.6f} "
          f"at epoch {best_epoch}")


def save_trace_plot(path, time_ms, truth, prediction, nodes, at_true, at_pred):
    figure, axes = plt.subplots(1, len(nodes), figsize=(4 * len(nodes), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, node in zip(axes, nodes):
        axis.plot(time_ms, truth[node], "k-", lw=2, label="GT")
        axis.plot(time_ms, prediction[node], "C3--", lw=1.5, label="prediction")
        axis.set_title(f"node {node}\nAT {at_true[node]:.1f}/{at_pred[node]:.1f} ms", fontsize=9)
        axis.set_xlabel("time (ms)"); axis.grid(alpha=0.3)
    axes[0].set_ylabel("V_m (mV)"); axes[0].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def evaluate(args, device):
    if not args.model_path:
        args.model_path = os.path.join("CheckPts", default_stem(args) + ".pt")
    raw_checkpoint = torch.load(args.model_path, map_location=device)
    config = raw_checkpoint["config"]
    model = FeatureDeepONet(**config).to(device)
    model.load_state_dict(raw_checkpoint["model_state_dict"]); model.eval()
    normalizer = FeatureNormalizer.from_state(raw_checkpoint["normalizer"])
    data = load_feature_data(args.data)
    output_dim = config["output_dim"]
    targets = data["targets"][:, :, :output_dim]
    train_idx, val_idx, test_idx = split_indices(len(data["theta"]),
                                                 raw_checkpoint["n_train"],
                                                 raw_checkpoint["n_val"])
    del train_idx, val_idx
    coords_tensor = tensor(normalizer.coords(data["coords"]), device)
    theta_test = tensor(normalizer.theta(data["theta"][test_idx]), device)

    predictions, infer_times = [], []
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.no_grad():
        for position in range(len(test_idx)):
            sync(); start = time.perf_counter()
            output = model(theta_test[position:position + 1], coords_tensor)
            sync(); infer_times.append(time.perf_counter() - start)
            predictions.append(to_numpy(output[0]))
    prediction = normalizer.targets_inverse(np.stack(predictions)).astype(np.float32)
    truth_features = targets[test_idx]
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    out_dir = args.out_dir or os.path.join("Predictions", stem, "Test")
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    def emit(message=""):
        print(message); lines.append(message)
    emit(f"model: {config} | {device}")
    emit(f"inference: {np.mean(infer_times) * 1000:.1f} +/- "
         f"{np.std(infer_times) * 1000:.1f} ms/case")
    emit(f"feature MAE on {len(test_idx)} test hearts:")
    for channel, name in enumerate(data["target_names"][:output_dim]):
        per_case = np.abs(prediction[..., channel] - truth_features[..., channel]).mean(axis=1)
        unit = " ms" if channel == 0 else ""
        emit(f"  {name:>20}: {per_case.mean():.5f} +/- {per_case.std():.5f}{unit}")
    direct_at_rel, direct_at_mae = [], []
    for position in range(len(test_idx)):
        rel, mae = at_metrics(prediction[position, :, 0], truth_features[position, :, 0])
        direct_at_rel.append(rel); direct_at_mae.append(mae)
    emit(f"direct AT: Rel L2 {np.mean(direct_at_rel):.5f} +/- {np.std(direct_at_rel):.5f} | "
         f"MAE {np.mean(direct_at_mae):.3f} +/- {np.std(direct_at_mae):.3f} ms")
    np.savez_compressed(os.path.join(out_dir, "test_features.npz"),
                        pred=prediction, true=truth_features,
                        target_names=data["target_names"][:output_dim], test_indices=test_idx,
                        case_names=(data["case_names"][test_idx]
                                    if data["case_names"] is not None else test_idx))

    if not args.skip_vm_eval:
        n_components = output_dim - 1
        basis = load_decoder_basis(args.basis, n_components)
        vm_archive = np.load(args.vm_data, allow_pickle=True)
        vm_all = vm_archive["vm"]  # ~15 GB decompressed once, no second copy
        time_ms = vm_archive["time"].astype(np.float32)
        if not np.array_equal(time_ms, basis["time"]):
            raise SystemExit("raw V_m and decoder basis use different time grids")
        dt = float(np.median(np.diff(time_ms)))
        vm_rel, vm_mae, decoded_at_mae, slope_fraction = [], [], [], []
        for position, global_case in enumerate(test_idx):
            decoded = decode_features(prediction[position], basis, args.chunk_nodes)
            truth = vm_all[global_case]
            rel, mae = vm_metrics(decoded, truth)
            vm_rel.append(rel); vm_mae.append(mae)
            at_decoded = activation_time(decoded, time_ms, basis["at_threshold"])
            _, at_mae_value = at_metrics(at_decoded, truth_features[position, :, 0])
            decoded_at_mae.append(at_mae_value)
            slope_fraction.append(median_max_dvdt(decoded, dt) / median_max_dvdt(truth, dt))
            emit(f"case {global_case:3d}: Vm RelL2 {rel:.5f} | MAE {mae:.3f} mV | "
                 f"decoded AT MAE {at_mae_value:.3f} ms | dVdt frac {slope_fraction[-1]:.3f}")
            if not args.skip_plots and position < args.n_viz:
                true_at = truth_features[position, :, 0]
                active = np.where(np.isfinite(true_at))[0]
                ordered = active[np.argsort(true_at[active])]
                nodes = ordered[np.linspace(0, len(ordered) - 1, 4).astype(int)]
                save_trace_plot(os.path.join(out_dir, f"case{global_case}_traces.png"),
                                time_ms, truth, decoded, nodes, true_at, at_decoded)
            del decoded
        emit("")
        emit(f"decoded V_m: Rel L2 {np.mean(vm_rel):.5f} +/- {np.std(vm_rel):.5f} | "
             f"MAE {np.mean(vm_mae):.3f} +/- {np.std(vm_mae):.3f} mV")
        emit(f"decoded AT MAE: {np.mean(decoded_at_mae):.3f} +/- "
             f"{np.std(decoded_at_mae):.3f} ms")
        emit(f"decoded upstroke fraction: {np.mean(slope_fraction):.3f} +/- "
             f"{np.std(slope_fraction):.3f}")
        np.savetxt(os.path.join(out_dir, "vm_metrics.csv"),
                   np.column_stack((test_idx, vm_rel, vm_mae, decoded_at_mae, slope_fraction)),
                   delimiter=",", comments="",
                   header="case,vm_rel_l2,vm_mae_mv,decoded_at_mae_ms,dvdt_fraction")
    else:
        emit("V_m decoding skipped (--skip-vm-eval)")

    with open(os.path.join(out_dir, "test_summary.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    emit(f"outputs -> {out_dir}")


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.test_model:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
