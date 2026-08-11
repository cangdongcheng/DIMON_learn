"""
Does a trained Geo-DONet behave like a temporal low-pass filter?
================================================================

For each requested Gaussian width, smooth the full-resolution f601 ground-truth
V_m only along time, sample it at the prediction time points, and compare it
with a saved neural-network prediction. By default, only the first test case is
processed; use ``--num-cases 0`` to process every saved test case.

The global table reports:

  * prediction vs temporally blurred ground truth: total absolute/squared
    error, MAE, RMSE, and relative L2;
  * blurred vs original ground truth: the distortion introduced by smoothing;
  * MAE improvement relative to the unblurred (sigma=0) comparison.

The first predicted case is exported as a ParaView VTU/PVD series using the
globally best sigma (minimum prediction-vs-blur MSE). Each frame contains:

  Vm_pred, Vm_original, Vm_temporal_blur,
  abs_err_original, abs_err_temporal_blur, and blur_delta.

Gaussian widths are standard deviations in milliseconds (FWHM = 2.355*sigma).
Smoothing uses nearest-value boundary padding and never mixes spatial nodes.

Run on a CPU node with >=32 GB memory. The compressed f601 archive expands to
~15 GB, while the saved f121 prediction contributes another ~0.6 GB.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_PREDICTIONS = (
    "/home/svu/e1032484/DIMON_learn/Geo_DONet/Predictions/"
    "geodonet_w300_d4_5000ep_lrsched/predictions.npz"
)
DEFAULT_GROUND_TRUTH = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
DEFAULT_MESH = "/home/svu/e1032484/scratch/canonical.vtu"
DEFAULT_OUT = "/home/svu/e1032484/scratch/temporal_blur_diagnostic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                        help="saved predictions.npz containing pred, time, case_names")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH,
                        help="full-resolution npz containing vm, time, case_names")
    parser.add_argument("--mesh", default=DEFAULT_MESH,
                        help="canonical-order tetrahedral VTU")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--num-cases", type=int, default=1,
                        help="number of saved test cases to process (default: 1; 0 means all)")
    parser.add_argument("--sigma-ms", type=float, nargs="+",
                        default=[0, 0.5, 1, 2, 3, 4, 5, 7.5, 10],
                        help="Gaussian standard deviations in milliseconds")
    parser.add_argument("--truncate", type=float, default=4.0,
                        help="Gaussian kernel radius in standard deviations")
    parser.add_argument("--mode", choices=("nearest", "reflect", "mirror"),
                        default="nearest", help="temporal boundary handling")
    parser.add_argument("--skip-vtu", action="store_true")
    parser.add_argument("--vtu-frame-step", type=int, default=1,
                        help="write every K-th prediction frame")
    return parser.parse_args()


def decode_names(values):
    return [str(value.decode() if isinstance(value, bytes) else value) for value in values]


def match_cases(pred_names, truth_names):
    lookup = {name: index for index, name in enumerate(truth_names)}
    missing = [name for name in pred_names if name not in lookup]
    if missing:
        raise SystemExit(f"ground-truth archive is missing prediction cases: {missing[:5]}")
    return np.asarray([lookup[name] for name in pred_names], dtype=np.int64)


def match_times(pred_time, truth_time):
    indices = np.searchsorted(truth_time, pred_time)
    indices = np.clip(indices, 0, len(truth_time) - 1)
    left = np.clip(indices - 1, 0, len(truth_time) - 1)
    choose_left = np.abs(truth_time[left] - pred_time) < np.abs(truth_time[indices] - pred_time)
    indices[choose_left] = left[choose_left]
    max_error = float(np.max(np.abs(truth_time[indices] - pred_time)))
    dt = float(np.median(np.diff(truth_time)))
    if max_error > dt * 1e-3 + 1e-5:
        raise SystemExit(f"prediction times do not lie on the ground-truth grid "
                         f"(maximum mismatch {max_error:g} ms)")
    return indices


def smooth_temporal(vm, sigma_frames, mode, truncate):
    if sigma_frames == 0:
        return vm
    return gaussian_filter1d(vm, sigma=sigma_frames, axis=1,
                             mode=mode, truncate=truncate, output=np.float32)


def empty_accumulator():
    return dict(pred_abs=0.0, pred_sq=0.0, blur_abs=0.0, blur_sq=0.0,
                target_sq=0.0, count=0)


def update_metrics(accumulator, prediction, original, blurred):
    pred_difference = prediction.astype(np.float64) - blurred
    blur_difference = blurred.astype(np.float64) - original
    accumulator["pred_abs"] += np.abs(pred_difference).sum(dtype=np.float64)
    accumulator["pred_sq"] += np.square(pred_difference).sum(dtype=np.float64)
    accumulator["blur_abs"] += np.abs(blur_difference).sum(dtype=np.float64)
    accumulator["blur_sq"] += np.square(blur_difference).sum(dtype=np.float64)
    accumulator["target_sq"] += np.square(blurred.astype(np.float64)).sum(dtype=np.float64)
    accumulator["count"] += prediction.size


def final_metrics(accumulator):
    count = accumulator["count"]
    return dict(
        total_abs_error=accumulator["pred_abs"],
        total_squared_error=accumulator["pred_sq"],
        mae=accumulator["pred_abs"] / count,
        rmse=np.sqrt(accumulator["pred_sq"] / count),
        rel_l2=np.sqrt(accumulator["pred_sq"] / (accumulator["target_sq"] + 1e-30)),
        blur_original_total_abs=accumulator["blur_abs"],
        blur_original_total_squared=accumulator["blur_sq"],
        blur_original_mae=accumulator["blur_abs"] / count,
        blur_original_rmse=np.sqrt(accumulator["blur_sq"] / count),
    )


def write_vtu_series(case_dir, points, tetra, time_ms, fields, frame_step):
    series_dir = os.path.join(case_dir, "series")
    os.makedirs(series_dir, exist_ok=True)
    rows = []
    cells = [("tetra", tetra)]
    output_frame = 0
    for frame in range(0, len(time_ms), frame_step):
        filename = f"frame_{output_frame:04d}.vtu"
        point_data = {name: np.asarray(values[:, frame], dtype=np.float32)
                      for name, values in fields.items()}
        meshio.write_points_cells(os.path.join(series_dir, filename), points, cells,
                                  point_data=point_data, binary=True)
        rows.append(f'    <DataSet timestep="{float(time_ms[frame])}" group="" part="0" '
                    f'file="series/{filename}"/>')
        output_frame += 1
    document = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *rows,
        '  </Collection>',
        '</VTKFile>',
    ]
    with open(os.path.join(case_dir, "Vm_temporal_blur.pvd"), "w") as handle:
        handle.write("\n".join(document) + "\n")


def save_error_plot(rows, path):
    sigma = np.asarray([row["sigma_ms"] for row in rows])
    mae = np.asarray([row["mae"] for row in rows])
    rel_l2 = np.asarray([row["rel_l2"] for row in rows])
    distortion = np.asarray([row["blur_original_mae"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(sigma, mae, "o-", label="prediction vs blurred GT")
    axes[0].plot(sigma, distortion, "s--", label="blurred vs original GT")
    axes[0].set_ylabel("MAE (mV)")
    axes[0].legend()
    axes[1].plot(sigma, rel_l2, "o-", color="C3")
    axes[1].set_ylabel("prediction vs blurred GT Rel L2")
    for axis in axes:
        axis.set_xlabel("temporal Gaussian sigma (ms)")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_first_case_traces(path, time_ms, prediction, original, blurred):
    # Nodes spanning the prediction's activation order, estimated by max dV/dt.
    activation_index = np.argmax(np.gradient(original, time_ms, axis=1), axis=1)
    order = np.argsort(activation_index)
    nodes = order[np.linspace(0, len(order) - 1, 4).astype(int)]
    figure, axes = plt.subplots(1, len(nodes), figsize=(4 * len(nodes), 3.5), sharey=True)
    for axis, node in zip(axes, nodes):
        axis.plot(time_ms, original[node], "k-", lw=2, label="original GT")
        axis.plot(time_ms, blurred[node], "C0--", lw=1.5, label="temporal blur")
        axis.plot(time_ms, prediction[node], "C3-", lw=1.2, label="network")
        axis.set_title(f"node {node}")
        axis.set_xlabel("time (ms)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("V_m (mV)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    args = parse_args()
    if args.num_cases < 0:
        raise SystemExit("--num-cases must be >= 0")
    if args.vtu_frame_step < 1:
        raise SystemExit("--vtu-frame-step must be >= 1")
    sigma_values = sorted(set(float(value) for value in args.sigma_ms))
    if not sigma_values or sigma_values[0] < 0:
        raise SystemExit("--sigma-ms values must be non-negative")
    if 0.0 not in sigma_values:
        sigma_values.insert(0, 0.0)
    os.makedirs(args.out, exist_ok=True)

    print(f"predictions : {args.predictions}", flush=True)
    pred_archive = np.load(args.predictions, allow_pickle=True)
    required = {"pred", "time", "case_names"}
    missing = required.difference(pred_archive.files)
    if missing:
        raise SystemExit(f"prediction archive is missing {sorted(missing)}")
    prediction = pred_archive["pred"].astype(np.float32)
    pred_time = pred_archive["time"].astype(np.float32)
    pred_names = decode_names(pred_archive["case_names"])
    if args.num_cases:
        prediction = prediction[:args.num_cases]
        pred_names = pred_names[:args.num_cases]
    if not pred_names:
        raise SystemExit("no prediction cases were selected")

    print(f"ground truth: {args.ground_truth}", flush=True)
    truth_archive = np.load(args.ground_truth, allow_pickle=True)
    truth_vm = truth_archive["vm"]  # f601 expands to ~15 GB; do not make a copy
    truth_time = truth_archive["time"].astype(np.float32)
    truth_names = decode_names(truth_archive["case_names"])
    case_indices = match_cases(pred_names, truth_names)
    time_indices = match_times(pred_time, truth_time)
    truth_dt = float(np.median(np.diff(truth_time)))

    if prediction.shape != (len(pred_names), truth_vm.shape[1], len(pred_time)):
        raise SystemExit(f"prediction shape {prediction.shape} is inconsistent with "
                         f"{len(pred_names)} cases, {truth_vm.shape[1]} nodes, "
                         f"{len(pred_time)} times")
    print(f"data         : {len(pred_names)} cases, {truth_vm.shape[1]} nodes; "
          f"GT dt={truth_dt:g} ms -> prediction dt={np.median(np.diff(pred_time)):g} ms",
          flush=True)
    print("sigmas (ms)  : " + ", ".join(f"{value:g}" for value in sigma_values), flush=True)

    accumulators = {sigma: empty_accumulator() for sigma in sigma_values}
    per_case_rows = []
    start_time = time.time()
    for position, global_case in enumerate(case_indices):
        original_full = truth_vm[global_case]
        original_eval = original_full[:, time_indices]
        for sigma in sigma_values:
            smoothed_full = smooth_temporal(original_full, sigma / truth_dt,
                                            args.mode, args.truncate)
            blurred_eval = smoothed_full[:, time_indices]
            local = empty_accumulator()
            update_metrics(local, prediction[position], original_eval, blurred_eval)
            update_metrics(accumulators[sigma], prediction[position], original_eval, blurred_eval)
            result = final_metrics(local)
            per_case_rows.append(dict(case_position=position,
                                      global_case=int(global_case),
                                      case_name=pred_names[position],
                                      sigma_ms=sigma,
                                      mae=result["mae"],
                                      rmse=result["rmse"],
                                      rel_l2=result["rel_l2"],
                                      blur_original_mae=result["blur_original_mae"]))
        elapsed = time.time() - start_time
        eta = elapsed / (position + 1) * (len(case_indices) - position - 1)
        print(f"case {position + 1:2d}/{len(case_indices)} ({pred_names[position]}) | "
              f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)

    rows = []
    baseline_mae = final_metrics(accumulators[0.0])["mae"]
    for sigma in sigma_values:
        row = dict(sigma_ms=sigma, sigma_frames=sigma / truth_dt,
                   **final_metrics(accumulators[sigma]))
        row["mae_improvement_percent"] = 100.0 * (baseline_mae - row["mae"]) / baseline_mae
        rows.append(row)
    best = min(rows, key=lambda row: row["total_squared_error"])

    table_path = os.path.join(args.out, "metrics_vs_sigma.csv")
    with open(table_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_case_path = os.path.join(args.out, "per_case_metrics.csv")
    with open(per_case_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0]))
        writer.writeheader()
        writer.writerows(per_case_rows)

    lines = [
        "=== temporal blur diagnostic ===",
        f"predictions: {args.predictions}",
        f"ground truth: {args.ground_truth}",
        f"cases/nodes/frames: {len(pred_names)} / {truth_vm.shape[1]} / {len(pred_time)}",
        f"GT dt / prediction dt: {truth_dt:g} / {float(np.median(np.diff(pred_time))):g} ms",
        "",
        (f"{'sigma(ms)':>9} {'MAE':>9} {'RMSE':>9} {'RelL2':>9} "
         f"{'blur-GT MAE':>12} {'MAE improve':>12}"),
    ]
    for row in rows:
        lines.append(f"{row['sigma_ms']:9.3f} {row['mae']:9.4f} {row['rmse']:9.4f} "
                     f"{row['rel_l2']:9.4f} {row['blur_original_mae']:12.4f} "
                     f"{row['mae_improvement_percent']:11.2f}%")
    lines.extend([
        "",
        f"best sigma by global MSE: {best['sigma_ms']:g} ms "
        f"(FWHM {2.35482 * best['sigma_ms']:.3f} ms)",
        f"baseline sigma=0 MAE: {baseline_mae:.6f} mV",
        f"best blurred-GT MAE: {best['mae']:.6f} mV",
        f"MAE improvement: {best['mae_improvement_percent']:.3f}%",
        f"best total absolute error: {best['total_abs_error']:.8e} mV",
        f"best total squared error: {best['total_squared_error']:.8e} mV^2",
    ])
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")
    save_error_plot(rows, os.path.join(args.out, "error_vs_sigma.png"))

    if not args.skip_vtu:
        mesh = meshio.read(args.mesh)
        tetra = next((block.data for block in mesh.cells if block.type == "tetra"), None)
        if tetra is None:
            raise SystemExit(f"{args.mesh} contains no tetrahedral cells")
        if mesh.points.shape[0] != truth_vm.shape[1]:
            raise SystemExit(f"mesh has {mesh.points.shape[0]} nodes, data has {truth_vm.shape[1]}")
        first_original_full = truth_vm[case_indices[0]]
        first_original = first_original_full[:, time_indices]
        first_blurred_full = smooth_temporal(first_original_full,
                                             best["sigma_ms"] / truth_dt,
                                             args.mode, args.truncate)
        first_blurred = first_blurred_full[:, time_indices]
        first_prediction = prediction[0]
        fields = {
            "Vm_pred": first_prediction,
            "Vm_original": first_original,
            "Vm_temporal_blur": first_blurred,
            "abs_err_original": np.abs(first_prediction - first_original),
            "abs_err_temporal_blur": np.abs(first_prediction - first_blurred),
            "blur_delta": first_blurred - first_original,
        }
        case_dir = os.path.join(args.out, "vtu",
                                f"{pred_names[0]}_sigma{best['sigma_ms']:g}ms")
        write_vtu_series(case_dir, mesh.points.astype(np.float32),
                         tetra.astype(np.int32), pred_time, fields,
                         args.vtu_frame_step)
        save_first_case_traces(os.path.join(args.out, "first_case_traces.png"),
                               pred_time, first_prediction, first_original, first_blurred)
        print(f"VTU/PVD      : {case_dir}/Vm_temporal_blur.pvd")
    print(f"metrics      : {table_path}")
    print(f"summary/plot : {args.out}/summary.txt, error_vs_sigma.png")


if __name__ == "__main__":
    main()
