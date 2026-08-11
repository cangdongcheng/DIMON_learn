"""
Test whether a trained Geo-DONet behaves like a spatial low-pass filter.

At every saved time frame, V_m is averaged only between nodes connected by an
edge of the canonical tetrahedral mesh. Time frames and hearts are never mixed.
Repeated relaxed neighbour-averaging passes produce progressively wider spatial
smoothing:

    V_next = (1 - alpha) * V + alpha * mean_mesh_neighbours(V)

The default is the first saved test case. Use ``--num-cases 0`` to evaluate all
saved test cases. The saved Geo-DONet archive already contains the matching
f121 ground truth, so the much larger f601 archive is not needed.

The reported RMS radius is only a scale guide:

    RMS radius ~= RMS mesh-edge length * sqrt(alpha * passes)

It is expressed in the same coordinate units as canonical.vtu.
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
from scipy.sparse import coo_matrix, triu


DEFAULT_PREDICTIONS = (
    "/home/svu/e1032484/DIMON_learn/Geo_DONet/Predictions/"
    "geodonet_w300_d4_5000ep_lrsched/predictions.npz"
)
DEFAULT_MESH = "/home/svu/e1032484/scratch/canonical.vtu"
DEFAULT_OUT = "/home/svu/e1032484/scratch/spatial_blur_diagnostic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                        help="Geo-DONet predictions.npz containing pred and true")
    parser.add_argument("--mesh", default=DEFAULT_MESH,
                        help="canonical-order tetrahedral VTU")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--num-cases", type=int, default=1,
                        help="number of saved test cases (default: 1; 0 means all)")
    parser.add_argument("--passes", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 16, 32, 64],
                        help="numbers of relaxed neighbour-averaging passes")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="neighbour contribution per pass; must be in (0, 1]")
    parser.add_argument("--skip-vtu", action="store_true")
    parser.add_argument("--vtu-frame-step", type=int, default=1,
                        help="write every K-th saved time frame")
    return parser.parse_args()


def decode_names(values):
    return [str(value.decode() if isinstance(value, bytes) else value)
            for value in values]


def tetrahedra_from_mesh(mesh):
    blocks = [block.data for block in mesh.cells if block.type == "tetra"]
    if not blocks:
        raise SystemExit("canonical mesh contains no tetrahedral cells")
    return np.concatenate(blocks, axis=0).astype(np.int32, copy=False)


def build_neighbour_operator(points, tetrahedra):
    """Return row-stochastic mesh-neighbour averaging and edge statistics."""
    left = tetrahedra[:, [0, 0, 0, 1, 1, 2]].reshape(-1)
    right = tetrahedra[:, [1, 2, 3, 2, 3, 3]].reshape(-1)
    rows = np.concatenate((left, right))
    cols = np.concatenate((right, left))
    values = np.ones(rows.size, dtype=np.float32)
    adjacency = coo_matrix(
        (values, (rows, cols)), shape=(len(points), len(points)),
        dtype=np.float32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data.fill(1.0)

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree == 0):
        isolated = np.flatnonzero(degree == 0)
        raise SystemExit(f"mesh contains {len(isolated)} isolated nodes")
    neighbour_mean = adjacency.multiply((1.0 / degree)[:, None]).tocsr()

    upper = triu(adjacency, k=1).tocoo()
    edge_delta = points[upper.row] - points[upper.col]
    edge_length = np.linalg.norm(edge_delta, axis=1)
    statistics = {
        "num_edges": int(len(edge_length)),
        "min_edge": float(np.min(edge_length)),
        "median_edge": float(np.median(edge_length)),
        "mean_edge": float(np.mean(edge_length)),
        "rms_edge": float(np.sqrt(np.mean(np.square(edge_length)))),
        "max_edge": float(np.max(edge_length)),
    }
    return neighbour_mean, statistics


def relaxed_average(values, neighbour_mean, alpha):
    averaged = neighbour_mean @ values
    if alpha == 1.0:
        return np.asarray(averaged, dtype=np.float32)
    return np.asarray((1.0 - alpha) * values + alpha * averaged,
                      dtype=np.float32)


def smooth_n_passes(values, neighbour_mean, alpha, passes):
    result = np.asarray(values, dtype=np.float32).copy()
    for _ in range(passes):
        result = relaxed_average(result, neighbour_mean, alpha)
    return result


def empty_accumulator():
    return {
        "pred_abs": 0.0,
        "pred_sq": 0.0,
        "smooth_abs": 0.0,
        "smooth_sq": 0.0,
        "target_sq": 0.0,
        "count": 0,
    }


def update_metrics(accumulator, prediction, original, smoothed):
    pred_difference = prediction.astype(np.float64) - smoothed
    smooth_difference = smoothed.astype(np.float64) - original
    accumulator["pred_abs"] += np.abs(pred_difference).sum(dtype=np.float64)
    accumulator["pred_sq"] += np.square(pred_difference).sum(dtype=np.float64)
    accumulator["smooth_abs"] += np.abs(smooth_difference).sum(dtype=np.float64)
    accumulator["smooth_sq"] += np.square(smooth_difference).sum(dtype=np.float64)
    accumulator["target_sq"] += np.square(
        smoothed.astype(np.float64)
    ).sum(dtype=np.float64)
    accumulator["count"] += prediction.size


def final_metrics(accumulator):
    count = accumulator["count"]
    return {
        "total_absolute_error": accumulator["pred_abs"],
        "total_squared_error": accumulator["pred_sq"],
        "mae": accumulator["pred_abs"] / count,
        "rmse": np.sqrt(accumulator["pred_sq"] / count),
        "rel_l2": np.sqrt(
            accumulator["pred_sq"] / (accumulator["target_sq"] + 1e-30)
        ),
        "smooth_original_mae": accumulator["smooth_abs"] / count,
        "smooth_original_rmse": np.sqrt(accumulator["smooth_sq"] / count),
    }


def write_vtu_series(case_dir, points, tetrahedra, time_ms, fields, frame_step):
    series_dir = os.path.join(case_dir, "series")
    os.makedirs(series_dir, exist_ok=True)
    datasets = []
    cells = [("tetra", tetrahedra)]
    output_frame = 0
    for frame in range(0, len(time_ms), frame_step):
        filename = f"frame_{output_frame:04d}.vtu"
        point_data = {
            name: np.asarray(values[:, frame], dtype=np.float32)
            for name, values in fields.items()
        }
        meshio.write_points_cells(
            os.path.join(series_dir, filename),
            points,
            cells,
            point_data=point_data,
            binary=True,
        )
        datasets.append(
            f'    <DataSet timestep="{float(time_ms[frame])}" group="" '
            f'part="0" file="series/{filename}"/>'
        )
        output_frame += 1

    document = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *datasets,
        "  </Collection>",
        "</VTKFile>",
    ]
    with open(os.path.join(case_dir, "Vm_spatial_blur.pvd"), "w") as handle:
        handle.write("\n".join(document) + "\n")


def save_error_plot(rows, path):
    radius = np.asarray([row["approx_rms_radius"] for row in rows])
    mae = np.asarray([row["mae"] for row in rows])
    rel_l2 = np.asarray([row["rel_l2"] for row in rows])
    distortion = np.asarray([row["smooth_original_mae"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(radius, mae, "o-", label="prediction vs smoothed GT")
    axes[0].plot(radius, distortion, "s--", label="smoothed vs original GT")
    axes[0].set_ylabel("MAE (mV)")
    axes[0].legend()
    axes[1].plot(radius, rel_l2, "o-", color="C3")
    axes[1].set_ylabel("prediction vs smoothed GT Rel L2")
    for axis in axes:
        axis.set_xlabel("approximate spatial RMS radius (mesh coordinate units)")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_trace_plot(path, time_ms, prediction, original, smoothed):
    activation_index = np.argmax(
        np.gradient(original, time_ms, axis=1), axis=1
    )
    order = np.argsort(activation_index)
    nodes = order[np.linspace(0, len(order) - 1, 4).astype(int)]
    figure, axes = plt.subplots(
        1, len(nodes), figsize=(4 * len(nodes), 3.5), sharey=True
    )
    for axis, node in zip(axes, nodes):
        axis.plot(time_ms, original[node], "k-", lw=2, label="original GT")
        axis.plot(time_ms, smoothed[node], "C0--", lw=1.5,
                  label="spatial blur")
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
    if not 0.0 < args.alpha <= 1.0:
        raise SystemExit("--alpha must be in (0, 1]")
    if args.vtu_frame_step < 1:
        raise SystemExit("--vtu-frame-step must be >= 1")
    passes = sorted(set(args.passes))
    if not passes or passes[0] < 0:
        raise SystemExit("--passes must contain non-negative integers")
    if 0 not in passes:
        passes.insert(0, 0)
    os.makedirs(args.out, exist_ok=True)

    print(f"predictions : {args.predictions}", flush=True)
    archive = np.load(args.predictions, allow_pickle=True)
    required = {"pred", "true", "time", "case_names"}
    missing = required.difference(archive.files)
    if missing:
        raise SystemExit(
            f"prediction archive is missing {sorted(missing)}; rerun Geo_DONet "
            "evaluation with ground truth and --save"
        )
    prediction = archive["pred"].astype(np.float32)
    original = archive["true"].astype(np.float32)
    time_ms = archive["time"].astype(np.float32)
    case_names = decode_names(archive["case_names"])
    if prediction.shape != original.shape:
        raise SystemExit(
            f"pred shape {prediction.shape} != true shape {original.shape}"
        )
    if prediction.ndim != 3 or prediction.shape[2] != len(time_ms):
        raise SystemExit(
            f"expected (cases, nodes, time), got {prediction.shape}"
        )
    if len(case_names) != prediction.shape[0]:
        raise SystemExit("case_names length does not match saved arrays")
    if args.num_cases:
        prediction = prediction[:args.num_cases]
        original = original[:args.num_cases]
        case_names = case_names[:args.num_cases]
    if not case_names:
        raise SystemExit("no test cases were selected")

    print(f"mesh        : {args.mesh}", flush=True)
    mesh = meshio.read(args.mesh)
    points = np.asarray(mesh.points[:, :3], dtype=np.float64)
    tetrahedra = tetrahedra_from_mesh(mesh)
    if len(points) != prediction.shape[1]:
        raise SystemExit(
            f"mesh has {len(points)} nodes, data has {prediction.shape[1]}"
        )
    print("building mesh-neighbour operator ...", flush=True)
    neighbour_mean, edge_stats = build_neighbour_operator(points, tetrahedra)
    bounds = np.ptp(points, axis=0)
    print(
        f"data        : {len(case_names)} case(s), {len(points)} nodes, "
        f"{len(time_ms)} frames",
        flush=True,
    )
    print(
        f"mesh        : {len(tetrahedra)} tetrahedra, "
        f"{edge_stats['num_edges']} unique edges",
        flush=True,
    )
    print(
        "mesh extent : " + " x ".join(f"{value:.6g}" for value in bounds)
        + " coordinate units",
        flush=True,
    )
    print(
        f"edge length : median {edge_stats['median_edge']:.6g}, "
        f"RMS {edge_stats['rms_edge']:.6g} coordinate units",
        flush=True,
    )
    print(f"passes      : {passes} | alpha={args.alpha:g}", flush=True)

    accumulators = {value: empty_accumulator() for value in passes}
    per_case_rows = []
    requested = set(passes)
    start_time = time.time()
    for case_position, case_name in enumerate(case_names):
        current = original[case_position].copy()
        for pass_index in range(passes[-1] + 1):
            if pass_index in requested:
                local = empty_accumulator()
                update_metrics(
                    local,
                    prediction[case_position],
                    original[case_position],
                    current,
                )
                update_metrics(
                    accumulators[pass_index],
                    prediction[case_position],
                    original[case_position],
                    current,
                )
                result = final_metrics(local)
                per_case_rows.append({
                    "case_position": case_position,
                    "case_name": case_name,
                    "passes": pass_index,
                    "approx_rms_radius": (
                        edge_stats["rms_edge"]
                        * np.sqrt(args.alpha * pass_index)
                    ),
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "rel_l2": result["rel_l2"],
                    "smooth_original_mae": result["smooth_original_mae"],
                })
            if pass_index < passes[-1]:
                current = relaxed_average(current, neighbour_mean, args.alpha)
        elapsed = time.time() - start_time
        eta = elapsed / (case_position + 1) * (
            len(case_names) - case_position - 1
        )
        print(
            f"case {case_position + 1}/{len(case_names)} ({case_name}) | "
            f"elapsed {elapsed / 60:.1f} min | eta {eta / 60:.1f} min",
            flush=True,
        )

    rows = []
    baseline_mae = final_metrics(accumulators[0])["mae"]
    for pass_count in passes:
        row = {
            "passes": pass_count,
            "alpha": args.alpha,
            "approx_rms_radius": (
                edge_stats["rms_edge"] * np.sqrt(args.alpha * pass_count)
            ),
            **final_metrics(accumulators[pass_count]),
        }
        row["mae_improvement_percent"] = (
            100.0 * (baseline_mae - row["mae"]) / baseline_mae
        )
        rows.append(row)
    best_mse = min(rows, key=lambda row: row["total_squared_error"])
    best_mae = min(rows, key=lambda row: row["mae"])

    table_path = os.path.join(args.out, "metrics_vs_passes.csv")
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
        "=== spatial blur diagnostic ===",
        f"predictions: {args.predictions}",
        f"mesh: {args.mesh}",
        (
            f"cases/nodes/frames: {len(case_names)} / "
            f"{len(points)} / {len(time_ms)}"
        ),
        f"neighbour relaxation alpha: {args.alpha:g}",
        (
            f"mesh edge median/RMS: {edge_stats['median_edge']:.6g} / "
            f"{edge_stats['rms_edge']:.6g} coordinate units"
        ),
        "",
        (
            f"{'passes':>7} {'RMS radius':>12} {'MAE':>9} {'RMSE':>9} "
            f"{'RelL2':>9} {'smooth-GT':>11} {'MAE improve':>12}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['passes']:7d} {row['approx_rms_radius']:12.5g} "
            f"{row['mae']:9.4f} {row['rmse']:9.4f} "
            f"{row['rel_l2']:9.4f} {row['smooth_original_mae']:11.4f} "
            f"{row['mae_improvement_percent']:11.2f}%"
        )
    lines.extend([
        "",
        (
            f"best by MSE: {best_mse['passes']} passes "
            f"(approx RMS radius {best_mse['approx_rms_radius']:.6g})"
        ),
        (
            f"best by MAE: {best_mae['passes']} passes "
            f"(approx RMS radius {best_mae['approx_rms_radius']:.6g})"
        ),
        f"baseline unsmoothed MAE: {baseline_mae:.6f} mV",
        f"best-MAE smoothed GT: {best_mae['mae']:.6f} mV",
        f"MAE improvement: {best_mae['mae_improvement_percent']:.3f}%",
        (
            "raw absolute-error sum at best MAE "
            f"(not normalized): {best_mae['total_absolute_error']:.8e} mV"
        ),
    ])
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")
    save_error_plot(rows, os.path.join(args.out, "error_vs_spatial_blur.png"))

    if not args.skip_vtu:
        best_passes = int(best_mae["passes"])
        first_original = original[0]
        first_smoothed = smooth_n_passes(
            first_original, neighbour_mean, args.alpha, best_passes
        )
        first_prediction = prediction[0]
        fields = {
            "Vm_pred": first_prediction,
            "Vm_original": first_original,
            "Vm_spatial_blur": first_smoothed,
            "abs_err_original": np.abs(first_prediction - first_original),
            "abs_err_spatial_blur": np.abs(
                first_prediction - first_smoothed
            ),
            "spatial_blur_delta": first_smoothed - first_original,
        }
        safe_name = case_names[0].replace("/", "_")
        case_dir = os.path.join(
            args.out, "vtu", f"{safe_name}_passes{best_passes}"
        )
        write_vtu_series(
            case_dir,
            points.astype(np.float32),
            tetrahedra,
            time_ms,
            fields,
            args.vtu_frame_step,
        )
        save_trace_plot(
            os.path.join(args.out, "first_case_traces.png"),
            time_ms,
            first_prediction,
            first_original,
            first_smoothed,
        )
        print(f"VTU/PVD      : {case_dir}/Vm_spatial_blur.pvd")
    print(f"metrics      : {table_path}")
    print(
        f"summary/plot : {args.out}/summary.txt, "
        "error_vs_spatial_blur.png"
    )


if __name__ == "__main__":
    main()
