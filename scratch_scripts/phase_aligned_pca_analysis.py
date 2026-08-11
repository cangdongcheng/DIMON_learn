"""
Phase-aligned lookup-table + temporal-PCA analysis of V_m(t).
================================================================

This is the oracle representation test for a phase-decoder surrogate:

    (theta, x) -> activation time tau + morphology coefficients a_k
    V_m(x, t)  = template_x(t - tau) + sum_k a_k phi_k(t - tau)

The sharp depolarisation is a transported feature.  Ordinary temporal PCA must
spend many modes representing the same upstroke at different activation times.
Here every waveform is first shifted so its -10 mV upward crossing occurs at a
common reference time.  We then build, using TRAIN hearts only:

  1. one global aligned mean waveform;
  2. a node-specific aligned lookup template (same canonical mesh node across
     hearts); and
  3. PCA modes of the aligned residual about the node template (default) or the
     global template (--pca-center global).

Evaluation uses the TRUE activation times and TRUE PCA coefficients of the 25
held-out test hearts.  These are deliberately oracle inputs: the result is the
best-case decoder/representation ceiling before training a network to predict
tau and a_k.  PCA and all templates remain train-only, so test V_m never leaks
into the decoder.

Outputs mirror pca_temporal_analysis.py:

  summary.txt                 human-readable results
  recon_vs_k.csv              aligned-PCA oracle metrics vs K
  template_baselines.csv      global/node lookup + shift round-trip floors
  explained_variance.png      residual PCA spectrum
  recon_vs_k.png              V_m/AT/upstroke metrics
  aligned_vs_unaligned.png    optional overlay with ordinary temporal PCA
  temporal_modes.png          aligned template and leading residual modes
  node_traces.png             early-to-late test-node waveform examples
  <basis-out>.npz             fixed decoder for the later NN experiment

The shift uses linear interpolation and constant edge padding.  The
align->unalign round-trip is reported because it is the irreducible numerical
floor of this decoder even with every PCA mode retained.

Run on a CPU compute node; see phase_aligned_pca.pbs.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Geo_DONet"))
from utils import load_dataset, split_indices, vm_rel_l2_mae, activation_time  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def shift_waveforms(waves, shift_ms, dt):
    """Sample each row at ``frame + shift_ms / dt`` using linear interpolation.

    ``waves`` is (n_waveforms, n_frames), ``shift_ms`` is (n_waveforms,).
    Values requested outside the time interval use constant edge padding.

    If a waveform activates at ``tau`` and the common reference is ``tau_ref``,
    alignment uses ``shift_ms=tau-tau_ref``.  Returning to physical time uses
    the opposite shift.
    """
    waves = np.asarray(waves, dtype=np.float32)
    shift_ms = np.asarray(shift_ms, dtype=np.float32)
    if waves.ndim != 2 or shift_ms.shape != (waves.shape[0],):
        raise ValueError("waves must be (n,T) and shift_ms must be (n,)")
    n_rows, n_frames = waves.shape
    if n_frames < 2:
        return waves.copy()

    position = (np.arange(n_frames, dtype=np.float32)[None, :]
                + shift_ms[:, None] / np.float32(dt))
    np.clip(position, 0.0, float(n_frames - 1), out=position)
    left = np.floor(position).astype(np.int32)
    np.minimum(left, n_frames - 2, out=left)
    fraction = position - left
    y0 = np.take_along_axis(waves, left, axis=1)
    y1 = np.take_along_axis(waves, left + 1, axis=1)
    return y0 + fraction * (y1 - y0)


def shifted_in_chunks(waves, shifts, dt, chunk_nodes):
    """Bounded-memory wrapper around shift_waveforms."""
    out = np.empty_like(waves, dtype=np.float32)
    for start in range(0, waves.shape[0], chunk_nodes):
        end = min(start + chunk_nodes, waves.shape[0])
        out[start:end] = shift_waveforms(waves[start:end], shifts[start:end], dt)
    return out


def max_dvdt(vm_2d, dt):
    """Same upstroke diagnostic used by pca_temporal_analysis.py."""
    return np.abs(np.gradient(vm_2d, dt, axis=1)).max(axis=1)


def case_metrics(recon, truth, true_at, time, threshold, dt):
    """Return scalar Rel-L2, MAE, AT MAE and median max-dV/dt for one heart."""
    rel_l2, mae = vm_rel_l2_mae(recon[None], truth[None])
    pred_at = activation_time(recon, time, threshold)
    valid = np.isfinite(pred_at) & np.isfinite(true_at)
    at_mae = float(np.abs(pred_at[valid] - true_at[valid]).mean()) if valid.any() else np.nan
    dvdt_med = float(np.median(max_dvdt(recon, dt)))
    return float(rel_l2[0]), float(mae[0]), at_mae, dvdt_med


def write_baseline_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "vm_rel_l2", "vm_mae", "at_mae_ms",
                         "dvdt_med", "dvdt_frac"])
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/svu/e1032484/scratch/geo_donet_data_f601.npz")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="subsample frames (1=f601/1ms, 2=f301/2ms, 5=f121/5ms)")
    ap.add_argument("--n-train", type=int, default=95)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--at-threshold", type=float, default=-10.0)
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100])
    ap.add_argument("--save-modes", type=int, default=100)
    ap.add_argument("--chunk-nodes", type=int, default=20_000,
                    help="waveforms shifted at once; lower this if memory is tight")
    ap.add_argument("--pca-center", choices=("node", "global"), default="node",
                    help="template subtracted before PCA (node lookup is the proposed decoder)")
    ap.add_argument("--out", default="/home/svu/e1032484/scratch/pca_phase_aligned")
    ap.add_argument("--basis-out",
                    default="/home/svu/e1032484/scratch/pca_phase_aligned_basis.npz")
    ap.add_argument("--unaligned-csv",
                    default="/home/svu/e1032484/scratch/pca_temporal/recon_vs_k.csv",
                    help="ordinary temporal-PCA CSV to overlay when present")
    ap.add_argument("--save-node-template", action=argparse.BooleanOptionalAction, default=True,
                    help="store the ~122 MB f601 node lookup table in --basis-out")
    args = ap.parse_args()

    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.basis_out) or ".", exist_ok=True)
    log_lines = []

    def log(message=""):
        print(message, flush=True)
        log_lines.append(message)

    log("=== phase-aligned lookup + temporal-PCA analysis ===")
    log(f"data         : {args.data}")
    d = load_dataset(args.data)
    vm = d["vm"]
    time = d["time"]
    if args.frame_step > 1:
        vm = vm[:, :, ::args.frame_step]
        time = time[::args.frame_step]
    n_cases, n_nodes, n_frames = vm.shape
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-5, atol=1e-6):
        raise SystemExit("phase alignment currently requires a uniformly sampled time grid")
    log(f"vm shape     : {vm.shape}  (C hearts, N nodes, T frames)")
    log(f"time         : {time[0]:.1f}..{time[-1]:.1f} ms, dt={dt:.3f} ms, "
        f"frame_step={args.frame_step}")
    log(f"vm range     : [{vm.min():.2f}, {vm.max():.2f}] mV")

    train_idx, val_idx, test_idx = split_indices(n_cases, args.n_train, args.n_val)
    log(f"split        : {len(train_idx)} train / {len(val_idx)} val / "
        f"{len(test_idx)} test (by heart)")

    # ---- training ATs: threshold crossing, computed one heart at a time ----
    train_at = []
    for case in train_idx:
        train_at.append(activation_time(vm[case], time, args.at_threshold))
    train_at = np.stack(train_at)
    valid_train = np.isfinite(train_at)
    if not valid_train.any():
        raise SystemExit(f"no training waveform crosses {args.at_threshold:g} mV")
    reference_at = float(np.median(train_at[valid_train]))
    log(f"alignment    : first upward crossing of {args.at_threshold:g} mV -> "
        f"reference {reference_at:.3f} ms")
    log(f"train AT     : [{np.nanmin(train_at):.3f}, {np.nanmax(train_at):.3f}] ms; "
        f"valid {valid_train.sum():,}/{valid_train.size:,} "
        f"({100 * valid_train.mean():.4f}%)")

    # ---- pass 1: train-only global and canonical-node lookup templates ----
    # float64 accumulation prevents visible averaging error over 95 hearts.
    global_sum = np.zeros(n_frames, dtype=np.float64)
    node_sum = np.zeros((n_nodes, n_frames), dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.int32)
    n_train_waveforms = 0
    log("building train-only aligned global + node lookup templates ...")
    for local_case, case in enumerate(train_idx):
        at = train_at[local_case]
        valid = np.isfinite(at)
        safe_at = np.where(valid, at, reference_at).astype(np.float32)
        shifts = safe_at - np.float32(reference_at)
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            local_valid = valid[start:end]
            if not local_valid.any():
                continue
            aligned = shift_waveforms(vm[case, start:end], shifts[start:end], dt)
            aligned = aligned[local_valid]
            node_ids = np.arange(start, end)[local_valid]
            global_sum += aligned.sum(axis=0, dtype=np.float64)
            node_sum[node_ids] += aligned
            node_count[node_ids] += 1
            n_train_waveforms += aligned.shape[0]
        if (local_case + 1) % 10 == 0 or local_case + 1 == len(train_idx):
            log(f"  templates: {local_case + 1}/{len(train_idx)} train hearts")

    global_template = (global_sum / n_train_waveforms).astype(np.float32)
    node_template = np.empty((n_nodes, n_frames), dtype=np.float32)
    has_node_template = node_count > 0
    node_template[has_node_template] = (
        node_sum[has_node_template] / node_count[has_node_template, None]).astype(np.float32)
    node_template[~has_node_template] = global_template
    del node_sum
    log(f"templates     : {n_train_waveforms:,} aligned waveforms; "
        f"node coverage {has_node_template.sum():,}/{n_nodes:,}")

    # ---- pass 2: PCA of residual aligned morphology ----
    # Stable blockwise Chan/Welford accumulation of the T x T covariance.
    residual_mean = np.zeros(n_frames, dtype=np.float64)
    residual_m2 = np.zeros((n_frames, n_frames), dtype=np.float64)
    residual_count = 0
    log(f"fitting residual PCA about the {args.pca_center} template ...")
    for local_case, case in enumerate(train_idx):
        at = train_at[local_case]
        valid = np.isfinite(at)
        safe_at = np.where(valid, at, reference_at).astype(np.float32)
        shifts = safe_at - np.float32(reference_at)
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            local_valid = valid[start:end]
            if not local_valid.any():
                continue
            aligned = shift_waveforms(vm[case, start:end], shifts[start:end], dt)
            if args.pca_center == "node":
                residual = aligned[local_valid] - node_template[start:end][local_valid]
            else:
                residual = aligned[local_valid] - global_template

            block = residual.astype(np.float64)
            block_count = block.shape[0]
            block_mean = block.mean(axis=0)
            block -= block_mean
            block_m2 = block.T @ block
            if residual_count == 0:
                residual_mean = block_mean
                residual_m2 = block_m2
                residual_count = block_count
            else:
                total = residual_count + block_count
                delta = block_mean - residual_mean
                residual_m2 += (block_m2
                                + np.outer(delta, delta)
                                * (residual_count * block_count / total))
                residual_mean += delta * (block_count / total)
                residual_count = total
        if (local_case + 1) % 10 == 0 or local_case + 1 == len(train_idx):
            log(f"  covariance: {local_case + 1}/{len(train_idx)} train hearts")

    covariance = residual_m2 / max(residual_count - 1, 1)
    eigval, eigvec = np.linalg.eigh(covariance)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0, None)
    components = eigvec[:, order].astype(np.float32)
    total_variance = eigval.sum()
    evr = eigval / total_variance if total_variance > 0 else np.zeros_like(eigval)
    cum_evr = np.cumsum(evr)
    residual_mean32 = residual_mean.astype(np.float32)
    del residual_m2, covariance

    def k_for(fraction):
        index = np.searchsorted(cum_evr, fraction)
        return int(index + 1) if index < n_frames else n_frames

    log("")
    log("--- aligned-residual explained variance ---")
    for fraction in (0.90, 0.95, 0.99, 0.999, 0.9999):
        log(f"  {fraction * 100:7.2f}% variance : K = {k_for(fraction)}")

    k_values = sorted(set(k for k in args.k_list if 0 < k <= n_frames))
    if not k_values:
        raise SystemExit("--k-list contains no values within the available frame count")
    k_max = max(k_values)

    # Metrics are accumulated per held-out heart to match Geo_DONet exactly.
    metric_names = ("vm_rel_l2", "vm_mae", "at_mae", "dvdt_med")
    pca_metrics = {k: {name: [] for name in metric_names} for k in k_values}
    baseline_metrics = {
        name: {metric: [] for metric in metric_names}
        for name in ("global_template", "node_template", "shift_roundtrip")
    }
    true_dvdt_medians = []
    trace_payload = None

    log("")
    log("evaluating oracle decoders on held-out test hearts ...")
    for test_position, case in enumerate(test_idx):
        truth = vm[case]
        true_at = activation_time(truth, time, args.at_threshold)
        valid = np.isfinite(true_at)
        safe_at = np.where(valid, true_at, reference_at).astype(np.float32)
        align_shift = safe_at - np.float32(reference_at)
        aligned = shifted_in_chunks(truth, align_shift, dt, args.chunk_nodes)
        unalign_shift = -align_shift
        true_dvdt_medians.append(float(np.median(max_dvdt(truth, dt))))

        global_aligned = np.broadcast_to(global_template, truth.shape)
        global_recon = shifted_in_chunks(global_aligned, unalign_shift, dt, args.chunk_nodes)
        node_recon = shifted_in_chunks(node_template, unalign_shift, dt, args.chunk_nodes)
        roundtrip = shifted_in_chunks(aligned, unalign_shift, dt, args.chunk_nodes)

        for name, recon in (("global_template", global_recon),
                            ("node_template", node_recon),
                            ("shift_roundtrip", roundtrip)):
            values = case_metrics(recon, truth, true_at, time, args.at_threshold, dt)
            for metric, value in zip(metric_names, values):
                baseline_metrics[name][metric].append(value)

        if args.pca_center == "node":
            base = node_template
        else:
            base = global_aligned
        centered = aligned - base - residual_mean32
        coefficients = centered @ components[:, :k_max]

        trace_recons = {}
        for k in k_values:
            aligned_recon = base + residual_mean32 + coefficients[:, :k] @ components[:, :k].T
            recon = shifted_in_chunks(aligned_recon, unalign_shift, dt, args.chunk_nodes)
            values = case_metrics(recon, truth, true_at, time, args.at_threshold, dt)
            for metric, value in zip(metric_names, values):
                pca_metrics[k][metric].append(value)
            if test_position == 0 and k in (5, 10, 20):
                trace_recons[k] = recon.copy()

        if test_position == 0:
            active = np.where(valid)[0]
            order_at = active[np.argsort(true_at[active])]
            picks = order_at[np.linspace(0, len(order_at) - 1, 4).astype(int)]
            trace_payload = dict(case=int(case), picks=picks, truth=truth[picks].copy(),
                                 true_at=true_at[picks].copy(),
                                 global_template=global_recon[picks].copy(),
                                 node_template=node_recon[picks].copy(),
                                 pca={k: value[picks].copy() for k, value in trace_recons.items()})

        log(f"  test: {test_position + 1}/{len(test_idx)} hearts")

    true_dvdt_med = float(np.median(true_dvdt_medians))

    # ---- aggregate and write the two result tables ----
    rows = []
    log("")
    log("--- aligned-PCA oracle reconstruction on TEST hearts ---")
    log(f"{'K':>5} {'cumEVR':>9} {'Vm RelL2':>9} {'Vm MAE':>8} {'AT MAE':>8} "
        f"{'dVdt med':>9} {'dVdt frac':>9}")
    for k in k_values:
        means = {name: float(np.nanmean(values))
                 for name, values in pca_metrics[k].items()}
        fraction = means["dvdt_med"] / true_dvdt_med
        rows.append((k, cum_evr[k - 1], means["vm_rel_l2"], means["vm_mae"],
                     means["at_mae"], means["dvdt_med"], fraction))
        log(f"{k:>5} {cum_evr[k - 1]:>9.5f} {means['vm_rel_l2']:>9.4f} "
            f"{means['vm_mae']:>8.3f} {means['at_mae']:>8.3f} "
            f"{means['dvdt_med']:>9.2f} {fraction:>9.3f}")
    rows = np.asarray(rows, dtype=np.float64)
    result_csv = os.path.join(args.out, "recon_vs_k.csv")
    np.savetxt(result_csv, rows, delimiter=",",
               header="K,cumEVR,vm_rel_l2,vm_mae,at_mae_ms,dvdt_med,dvdt_frac",
               comments="", fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])

    baseline_rows = []
    log("")
    log("--- oracle template/interpolation baselines on TEST hearts ---")
    log(f"{'method':>18} {'Vm RelL2':>9} {'Vm MAE':>8} {'AT MAE':>8} "
        f"{'dVdt med':>9} {'dVdt frac':>9}")
    for name in ("global_template", "node_template", "shift_roundtrip"):
        means = {metric: float(np.nanmean(values))
                 for metric, values in baseline_metrics[name].items()}
        fraction = means["dvdt_med"] / true_dvdt_med
        baseline_rows.append((name, means["vm_rel_l2"], means["vm_mae"],
                              means["at_mae"], means["dvdt_med"], fraction))
        log(f"{name:>18} {means['vm_rel_l2']:>9.4f} {means['vm_mae']:>8.3f} "
            f"{means['at_mae']:>8.3f} {means['dvdt_med']:>9.2f} {fraction:>9.3f}")
    log(f"  true median max dV/dt = {true_dvdt_med:.2f} mV/ms")
    baseline_csv = os.path.join(args.out, "template_baselines.csv")
    write_baseline_csv(baseline_csv, baseline_rows)
    log(f"saved tables  : {result_csv}, {baseline_csv}")

    # ---- fixed decoder artefact for the later network track ----
    n_save = min(args.save_modes, n_frames)
    basis_payload = dict(
        global_template=global_template,
        residual_mean=residual_mean32,
        components=components[:, :n_save],
        eigval=eigval[:n_save], evr=evr[:n_save], cum_evr=cum_evr[:n_save],
        time=time, reference_at=np.float32(reference_at),
        at_threshold=np.float32(args.at_threshold), n_train=np.int64(args.n_train),
        frame_step=np.int64(args.frame_step), pca_center=np.asarray(args.pca_center),
        shift_interpolation=np.asarray("linear_constant_edge"),
    )
    if args.save_node_template:
        basis_payload["node_template"] = node_template
    np.savez(args.basis_out, **basis_payload)
    log(f"saved basis   : {args.basis_out} ({n_save} modes; "
        f"node template {'included' if args.save_node_template else 'omitted'})")

    # ---- plots: keep styling and metric choices parallel to the old analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(np.arange(1, n_frames + 1), evr, ".-")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained var ratio")
    axes[0].set_title(f"aligned residual EVR ({args.pca_center} centre)")
    axes[0].set_xlim(0, min(n_frames, 200))
    axes[1].plot(np.arange(1, n_frames + 1), cum_evr, ".-")
    for fraction in (0.9, 0.99, 0.999):
        axes[1].axhline(fraction, color="grey", ls=":", lw=0.8)
    axes[1].set_xlabel("# components K")
    axes[1].set_ylabel("cumulative EVR")
    axes[1].set_title("aligned residual cumulative variance")
    axes[1].set_xlim(0, min(n_frames, 200)); axes[1].set_ylim(0.0, 1.001)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "explained_variance.png"), dpi=120)
    plt.close(fig)

    baseline_lookup = {row[0]: row for row in baseline_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_specs = ((2, "test V_m Rel L2"), (4, "test AT MAE (ms)"),
                  (6, "upstroke fidelity: recon/true dV/dt"))
    for axis, (column, title) in zip(axes, plot_specs):
        axis.semilogx(rows[:, 0], rows[:, column], "o-", label="aligned PCA oracle")
        base_column = {2: 1, 4: 3, 6: 5}[column]
        axis.axhline(baseline_lookup["global_template"][base_column], color="C1", ls="--",
                     label="global shifted template")
        axis.axhline(baseline_lookup["node_template"][base_column], color="C2", ls=":",
                     label="node shifted template")
        if column == 6:
            axis.axhline(1.0, color="grey", ls="-.", lw=0.8)
            axis.set_ylim(0, 1.1)
        axis.set_title(title); axis.set_xlabel("K"); axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "recon_vs_k.png"), dpi=120)
    plt.close(fig)

    phase = time - np.float32(reference_at)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(phase, global_template, "k-", lw=2, label="global aligned template")
    for k in range(min(6, n_frames)):
        axis.plot(phase, components[:, k] * np.sqrt(eigval[k]),
                  label=f"mode {k + 1} ({evr[k] * 100:.1f}%)")
    axis.axvline(0, color="grey", ls=":", lw=0.8)
    axis.set_xlabel("phase t - activation time (ms)")
    axis.set_ylabel("mV (mode scaled by sqrt eigval)")
    axis.set_title("activation-aligned template + residual PCA modes")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "temporal_modes.png"), dpi=120)
    plt.close(fig)

    if trace_payload is not None:
        fig, axes = plt.subplots(1, len(trace_payload["picks"]),
                                 figsize=(4 * len(trace_payload["picks"]), 3.8), sharey=True)
        for j, node in enumerate(trace_payload["picks"]):
            axis = axes[j]
            axis.plot(time, trace_payload["truth"][j], "k-", lw=2, label="GT")
            axis.plot(time, trace_payload["global_template"][j], color="C1", ls="--",
                      label="global template")
            axis.plot(time, trace_payload["node_template"][j], color="C2", ls=":",
                      label="node template")
            for k, traces in sorted(trace_payload["pca"].items()):
                axis.plot(time, traces[j], lw=1, label=f"aligned PCA K={k}")
            axis.set_title(f"node {node}, AT={trace_payload['true_at'][j]:.1f} ms", fontsize=9)
            axis.set_xlabel("time (ms)"); axis.grid(alpha=0.3)
            if j == 0:
                axis.set_ylabel("V_m (mV)"); axis.legend(fontsize=7)
        fig.suptitle(f"test heart {trace_payload['case']} — oracle phase-decoder reconstruction")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "node_traces.png"), dpi=120)
        plt.close(fig)

    if os.path.exists(args.unaligned_csv):
        ordinary = np.genfromtxt(args.unaligned_csv, delimiter=",", names=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        comparisons = (("vm_rel_l2", 2, "V_m Rel L2"),
                       ("at_mae_ms", 4, "AT MAE (ms)"),
                       ("dvdt_frac", 6, "upstroke dV/dt fraction"))
        for axis, (field, column, title) in zip(axes, comparisons):
            axis.semilogx(ordinary["K"], ordinary[field], "o-", color="C0",
                          label="ordinary temporal PCA")
            axis.semilogx(rows[:, 0], rows[:, column], "s-", color="C3",
                          label=f"phase-aligned PCA ({args.pca_center} centre)")
            if field == "dvdt_frac":
                axis.axhline(1.0, color="grey", ls=":", lw=0.8); axis.set_ylim(0, 1.1)
            axis.set_xlabel("K"); axis.set_title(title); axis.grid(alpha=0.3, which="both")
        axes[0].legend(fontsize=8)
        fig.suptitle("Ordinary vs activation-aligned temporal PCA — oracle test reconstruction")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(args.out, "aligned_vs_unaligned.png"), dpi=130)
        plt.close(fig)
        log(f"comparison    : overlaid ordinary PCA from {args.unaligned_csv}")
    else:
        log(f"comparison    : skipped; ordinary PCA CSV not found: {args.unaligned_csv}")

    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write("\n".join(log_lines) + "\n")
    log(f"plots         : {args.out}/")
    log("done.")


if __name__ == "__main__":
    main()
