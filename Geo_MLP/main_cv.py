"""Five-fold cross-validation for the vanilla geometry-conditioned MLP.

The fold layout exactly follows Geo_DONet/main_cv.py:

* shuffle all hearts once with NumPy RandomState(seed=42);
* split into five disjoint 25-heart test folds;
* from the other 100 hearts, use the last 5 for validation and 95 for fit;
* fit every normalization statistic using only those 95 hearts.

Training uses sampled (node, time) queries, as in Geo_MLP/main.py. Evaluation
reconstructs the complete field for every held-out heart. Each of the 125 hearts
therefore contributes exactly once to the pooled test result.
"""

import argparse
import csv
import os
import time as timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR

from main import (
    AT_THRESHOLD_MV,
    DATA_FILE,
    FRAME_STEP,
    GeoMLP,
    Normalizer,
    activation_time,
    at_rel_l2_mae,
    full_field_forward,
    load_strided_data,
    make_query,
    sampled_targets,
    set_seed,
    vm_rel_l2_mae,
)
from model import paired_forward


N_FOLDS = 5
N_VAL = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-fold CV for [geometry, coordinates, time] -> V_m MLP"
    )
    parser.add_argument("--data-path", default=DATA_FILE)
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-case", type=int, default=4096)
    parser.add_argument("--val-samples-per-case", type=int, default=16384)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-scheduled", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--patience", type=int, default=500,
                        help="epochs without validation improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-batch-size", type=int, default=131072)
    parser.add_argument("--query-frame-chunk", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def validate_args(args, n_cases=None):
    positive = {
        "--frame-step": args.frame_step,
        "--folds": args.folds,
        "--n-val": args.n_val,
        "--epochs": args.epochs,
        "--case-batch-size": args.case_batch_size,
        "--samples-per-case": args.samples_per_case,
        "--val-samples-per-case": args.val_samples_per_case,
        "--width": args.width,
        "--depth": args.depth,
        "--query-batch-size": args.query_batch_size,
        "--query-frame-chunk": args.query_frame_chunk,
        "--log-every": args.log_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise SystemExit(f"arguments must be positive: {', '.join(invalid)}")
    if args.patience < 0:
        raise SystemExit("--patience must be >= 0")
    if n_cases is not None:
        if n_cases % args.folds:
            raise SystemExit(
                f"{n_cases} cases cannot be divided evenly into {args.folds} folds"
            )
        remaining = n_cases - n_cases // args.folds
        if args.n_val >= remaining:
            raise SystemExit(
                f"--n-val {args.n_val} leaves no fitting cases per fold"
            )


def make_folds(n_cases, n_folds, n_val, seed):
    """Return the exact shuffled fold protocol used by Geo_DONet CV."""
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


def grid_activation_time(vm, time_ms, threshold=AT_THRESHOLD_MV):
    """Legacy Geo_DONet-CV definition: first saved frame above threshold."""
    crossed = vm > threshold
    first = np.argmax(crossed, axis=-1)
    activated = np.any(crossed, axis=-1)
    output = np.full(vm.shape[0], np.nan, dtype=np.float32)
    output[activated] = time_ms[first[activated]]
    return output


def activation_metrics(prediction, truth, time_ms, interpolated):
    if interpolated:
        pred_at = activation_time(prediction, time_ms, AT_THRESHOLD_MV)
        true_at = activation_time(truth, time_ms, AT_THRESHOLD_MV)
    else:
        pred_at = grid_activation_time(prediction, time_ms)
        true_at = grid_activation_time(truth, time_ms)
    rel_l2, mae = at_rel_l2_mae(pred_at[None, :], true_at[None, :])
    return float(rel_l2[0]), float(mae[0])


def save_loss_plot(history, path, fold):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(history["train"], label="train sampled MSE")
    axis.semilogy(history["val"], label="val fixed sampled MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized MSE")
    axis.set_title(f"Geo_MLP CV fold {fold}")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train_and_evaluate_fold(
    fold, train_idx, val_idx, test_idx,
    theta, coords, vm, time_ms, case_names, args, out_dir, device,
):
    set_seed(args.seed + fold)
    n_nodes, n_frames = vm.shape[1], vm.shape[2]
    fold_dir = os.path.join(out_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    normalizer = Normalizer(theta[train_idx], vm[train_idx], coords, time_ms)
    to_device = lambda values: torch.as_tensor(
        values, dtype=torch.float32, device=device
    )
    theta_train = to_device(normalizer.theta(theta[train_idx]))
    theta_val = to_device(normalizer.theta(theta[val_idx]))
    theta_test = to_device(normalizer.theta(theta[test_idx]))
    vm_train = to_device(normalizer.vm(vm[train_idx]))
    vm_val = to_device(normalizer.vm(vm[val_idx]))
    coords_norm = to_device(normalizer.coords(coords))
    time_norm = to_device(normalizer.time(time_ms))

    model = GeoMLP(
        geo_dim=theta.shape[1], coord_dim=coords.shape[1],
        width=args.width, depth=args.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        LinearLR(optimizer, 1.0, 0.1, args.epochs)
        if args.lr_scheduled else None
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1000 + fold)
    val_nodes = torch.randint(
        n_nodes, (args.val_samples_per_case,),
        generator=generator, device=device,
    )
    val_times = torch.randint(
        n_frames, (args.val_samples_per_case,),
        generator=generator, device=device,
    )
    val_query = make_query(coords_norm, time_norm, val_nodes, val_times)
    val_cases = torch.arange(len(val_idx), device=device)
    val_target = sampled_targets(vm_val, val_cases, val_nodes, val_times)

    checkpoint_path = os.path.join(fold_dir, "model.pt")
    loss_path = os.path.join(fold_dir, "loss.csv")
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    order = np.arange(len(train_idx))
    history = {"train": [], "val": []}
    fold_start = timer.time()

    with open(loss_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_sampled_mse", "val_fixed_sampled_mse"))
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(order)
            running = 0.0
            batches = 0
            for batch_start in range(0, len(order), args.case_batch_size):
                batch = torch.as_tensor(
                    order[batch_start:batch_start + args.case_batch_size],
                    dtype=torch.long, device=device,
                )
                nodes = torch.randint(
                    n_nodes, (args.samples_per_case,), device=device
                )
                times = torch.randint(
                    n_frames, (args.samples_per_case,), device=device
                )
                query = make_query(coords_norm, time_norm, nodes, times)
                target = sampled_targets(vm_train, batch, nodes, times)
                prediction = paired_forward(model, theta_train[batch], query)
                loss = torch.mean((prediction - target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                batches += 1
            if scheduler is not None:
                scheduler.step()
            train_loss = running / batches

            model.eval()
            with torch.no_grad():
                val_prediction = paired_forward(model, theta_val, val_query)
                val_loss = float(
                    torch.mean((val_prediction - val_target) ** 2).item()
                )
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            writer.writerow((epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"))

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "geo_dim": theta.shape[1],
                            "coord_dim": coords.shape[1],
                            "width": args.width,
                            "depth": args.depth,
                        },
                        "fold": fold,
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                        "test_idx": test_idx,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1

            if epoch % args.log_every == 0:
                handle.flush()
                elapsed = timer.time() - fold_start
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(
                    f"  Fold {fold} epoch {epoch}/{args.epochs}: "
                    f"train={train_loss:.6f} val={val_loss:.6f} "
                    f"best={best_val:.6f}@{best_epoch} "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )
            if args.patience > 0 and stale >= args.patience:
                print(
                    f"  Fold {fold} early stop at epoch {epoch}: "
                    f"no improvement for {args.patience} epochs",
                    flush=True,
                )
                break

    save_loss_plot(
        history, os.path.join(fold_dir, "loss.png"), fold
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = {
        "vm_l2": [], "vm_mae": [],
        "at_l2": [], "at_mae": [],
        "at_interp_l2": [], "at_interp_mae": [],
        "inference_time_s": [],
    }
    is_cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for position, global_case in enumerate(test_idx):
            if is_cuda:
                torch.cuda.synchronize()
            inference_start = timer.time()
            pred_normalized = full_field_forward(
                model,
                theta_test[position],
                coords_norm,
                time_norm,
                args.query_batch_size,
                args.query_frame_chunk,
            )
            if is_cuda:
                torch.cuda.synchronize()
            metrics["inference_time_s"].append(
                timer.time() - inference_start
            )
            prediction = normalizer.vm_inverse(pred_normalized).astype(np.float32)
            truth = vm[global_case]
            vm_l2, vm_mae = vm_rel_l2_mae(
                prediction[None, ...], truth[None, ...]
            )
            at_l2, at_mae = activation_metrics(
                prediction, truth, time_ms, interpolated=False
            )
            interp_l2, interp_mae = activation_metrics(
                prediction, truth, time_ms, interpolated=True
            )
            metrics["vm_l2"].append(float(vm_l2[0]))
            metrics["vm_mae"].append(float(vm_mae[0]))
            metrics["at_l2"].append(at_l2)
            metrics["at_mae"].append(at_mae)
            metrics["at_interp_l2"].append(interp_l2)
            metrics["at_interp_mae"].append(interp_mae)
            print(
                f"  Fold {fold} test {position + 1:2d}/{len(test_idx)} "
                f"{case_names[global_case]} | V_m MAE {vm_mae[0]:.3f} mV | "
                f"AT MAE {at_mae:.3f} ms",
                flush=True,
            )

    for key in metrics:
        metrics[key] = np.asarray(metrics[key], dtype=np.float32)
    fold_time_min = (timer.time() - fold_start) / 60.0

    del theta_train, theta_val, theta_test, vm_train, vm_val
    del coords_norm, time_norm, val_query, val_target
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        **metrics,
        "test_idx": np.asarray(test_idx),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "fold_time_min": fold_time_min,
    }


def mean_std(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), float("nan")
    return float(values[finite].mean()), float(values[finite].std())


def format_stat(values, decimals=4):
    mean, std = mean_std(values)
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def save_summary_plot(fold_results, out_path):
    metric_specs = [
        ("vm_l2", "V_m relative L2"),
        ("vm_mae", "V_m MAE (mV)"),
        ("at_l2", "AT relative L2 (legacy grid)"),
        ("at_mae", "AT MAE (ms; legacy grid)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    x = np.arange(len(fold_results))
    for axis, (key, label) in zip(axes.ravel(), metric_specs):
        fold_stats = [mean_std(result[key]) for result in fold_results]
        means = np.asarray([value[0] for value in fold_stats])
        stds = np.asarray([value[1] for value in fold_stats])
        valid = np.isfinite(means) & np.isfinite(stds)
        if valid.any():
            axis.errorbar(x[valid], means[valid], yerr=stds[valid],
                          fmt="o", capsize=4)
        pooled = np.concatenate([result[key] for result in fold_results])
        pooled_mean, _ = mean_std(pooled)
        if np.isfinite(pooled_mean):
            axis.axhline(pooled_mean, color="C3", linestyle="--",
                         label="pooled mean")
        axis.set_xticks(x)
        axis.set_xlabel("test fold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        if valid.any():
            axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_strided_data(args.data_path, args.frame_step)
    theta, coords, vm, time_ms = (
        data["theta"], data["coords"], data["vm"], data["time"]
    )
    case_names = (
        np.asarray([str(value) for value in data["case_names"]])
        if data["case_names"] is not None
        else np.asarray([f"case{i}" for i in range(len(theta))])
    )
    validate_args(args, n_cases=len(theta))
    shuffled_indices, folds = make_folds(
        len(theta), args.folds, args.n_val, args.seed
    )

    schedule_tag = "_lrsched" if args.lr_scheduled else ""
    default_dir = (
        f"CV_{args.folds}fold_{args.epochs}ep_w{args.width}_d{args.depth}_"
        f"s{args.samples_per_case}_f{vm.shape[2]}{schedule_tag}"
    )
    out_dir = args.out_dir or default_dir
    os.makedirs(out_dir, exist_ok=True)
    n_parameters = sum(
        parameter.numel()
        for parameter in GeoMLP(
            theta.shape[1], coords.shape[1], args.width, args.depth
        ).parameters()
    )

    print(f"Save dir: {out_dir}")
    print(
        f"Data: {len(theta)} hearts, {len(coords)} nodes, {vm.shape[2]} frames, "
        f"dt={np.median(np.diff(time_ms)):g} ms"
    )
    print(
        f"Model: GeoMLP w{args.width} d{args.depth}, "
        f"{n_parameters:,} parameters | {device}"
    )
    print(
        f"{args.folds}-fold CV, {args.epochs} epochs/fold, "
        f"samples={args.samples_per_case}, "
        f"val-samples={args.val_samples_per_case}, "
        f"lr-scheduled={args.lr_scheduled}, patience={args.patience}\n"
    )

    fold_results = []
    total_start = timer.time()
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"=== Fold {fold} ===")
        print(
            f"  Train: {len(train_idx)}, Val: {len(val_idx)}, "
            f"Test: {len(test_idx)}"
        )
        result = train_and_evaluate_fold(
            fold, train_idx, val_idx, test_idx,
            theta, coords, vm, time_ms, case_names,
            args, out_dir, device,
        )
        fold_results.append(result)
        print(
            f"  Fold {fold}: V_m L2={format_stat(result['vm_l2'])}, "
            f"MAE={format_stat(result['vm_mae'], 2)} mV | "
            f"AT L2={format_stat(result['at_l2'])}, "
            f"MAE={format_stat(result['at_mae'], 2)} ms | "
            f"best epoch {result['best_epoch']} | "
            f"{result['fold_time_min']:.1f} min\n"
        )

    pooled = {
        key: np.concatenate([result[key] for result in fold_results])
        for key in (
            "vm_l2", "vm_mae", "at_l2", "at_mae",
            "at_interp_l2", "at_interp_mae", "inference_time_s",
        )
    }
    all_test_idx = np.concatenate(
        [result["test_idx"] for result in fold_results]
    )
    fold_id = np.concatenate([
        np.full(len(result["test_idx"]), fold, dtype=np.int32)
        for fold, result in enumerate(fold_results)
    ])
    total_min = (timer.time() - total_start) / 60.0

    lines = [
        "=" * 68,
        f"{args.folds}-fold pooled (N={len(all_test_idx)} hearts):",
        f"  V_m  Rel L2 = {format_stat(pooled['vm_l2'])}",
        f"  V_m  MAE    = {format_stat(pooled['vm_mae'], 2)} mV",
        f"  AT grid Rel L2 = {format_stat(pooled['at_l2'])}",
        f"  AT grid MAE    = {format_stat(pooled['at_mae'], 2)} ms",
        f"  AT interpolated Rel L2 = {format_stat(pooled['at_interp_l2'])}",
        f"  AT interpolated MAE    = {format_stat(pooled['at_interp_mae'], 2)} ms",
        f"  Inference = {format_stat(pooled['inference_time_s'], 3)} s/heart",
        "",
        "Per-fold:",
    ]
    for fold, result in enumerate(fold_results):
        lines.append(
            f"  Fold {fold}: V_m L2={format_stat(result['vm_l2'])}, "
            f"MAE={format_stat(result['vm_mae'], 2)} mV, "
            f"AT MAE={format_stat(result['at_mae'], 2)} ms, "
            f"best epoch={result['best_epoch']}, "
            f"time={result['fold_time_min']:.1f} min"
        )
    lines.append(f"\nTotal time: {total_min:.1f} min")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")

    ordered_names = case_names[all_test_idx]
    with open(os.path.join(out_dir, "per_case_metrics.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "fold", "global_case", "case_name", "vm_rel_l2", "vm_mae_mv",
            "at_grid_rel_l2", "at_grid_mae_ms",
            "at_interp_rel_l2", "at_interp_mae_ms", "inference_time_s",
        ))
        for position in range(len(all_test_idx)):
            writer.writerow((
                int(fold_id[position]), int(all_test_idx[position]),
                ordered_names[position],
                pooled["vm_l2"][position], pooled["vm_mae"][position],
                pooled["at_l2"][position], pooled["at_mae"][position],
                pooled["at_interp_l2"][position],
                pooled["at_interp_mae"][position],
                pooled["inference_time_s"][position],
            ))

    np.savez_compressed(
        os.path.join(out_dir, "cv_results.npz"),
        case_names=ordered_names,
        test_idx=all_test_idx,
        fold_id=fold_id,
        fold_indices=shuffled_indices,
        vm_l2=pooled["vm_l2"],
        vm_mae=pooled["vm_mae"],
        at_l2=pooled["at_l2"],
        at_mae=pooled["at_mae"],
        at_interp_l2=pooled["at_interp_l2"],
        at_interp_mae=pooled["at_interp_mae"],
        inference_time_s=pooled["inference_time_s"],
        best_epoch=np.asarray([result["best_epoch"] for result in fold_results]),
        best_val_loss=np.asarray([
            result["best_val_loss"] for result in fold_results
        ]),
        fold_time_min=np.asarray([
            result["fold_time_min"] for result in fold_results
        ]),
    )
    save_summary_plot(
        fold_results, os.path.join(out_dir, "cv_summary.png")
    )
    print(f"Saved {out_dir}/cv_results.npz, per_case_metrics.csv, cv_summary.png")


if __name__ == "__main__":
    main()
