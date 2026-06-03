"""
Geo-DONet output layer — the rendering / IO that main.py's infer() calls.

Metric tables (text), V_m(t) traces, 3D V_m/AT scatter SVGs, standalone
colorbars, and predicted-V_m ParaView .vtu series. Kept separate so main.py
stays orchestration; no model or data logic here. Pulls in matplotlib / meshio /
multiprocessing, which is exactly why it is not in the light utils.py.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio


# colour scales shared across the V_m / AT renders (inherited from Geo_DONet/main.py)
VM_MIN, VM_MAX = -90.0, 50.0           # V_m (mV)
VM_ERR_MIN, VM_ERR_MAX = 0.0, 20.0     # |dV_m| (mV)
CMAP_VM = CMAP_AT = "RdYlBu_r"
CMAP_ERR = "Reds"


# ════════════════════ metric tables (text) ════════════════════
def format_metric_table(name, case_names, col1, col2, head1, head2):
    lines = [f"\n{name + ' metrics':<34}{head1:>10}{head2:>12}", "-" * 56]
    for c in range(len(col1)):
        label = str(case_names[c]) if case_names is not None else f"case{c}"
        lines.append(f"{label[:34]:<34}{col1[c]:>10.4f}{col2[c]:>12.2f}")
    lines.append("-" * 56)
    lines.append(f"{'mean':<34}{np.nanmean(col1):>10.4f}{np.nanmean(col2):>12.2f}")
    lines.append(f"{'std':<34}{np.nanstd(col1):>10.4f}{np.nanstd(col2):>12.2f}")
    return "\n".join(lines)


def format_metric_summary(n_cases, cases_name, vm_rel, vm_mae, at_rel, at_mae):
    """Compact mean ± std block — the headline result, easy to paste into slides."""
    def stat(values): return np.nanmean(values), np.nanstd(values)
    vr, vmae, ar, amae = stat(vm_rel), stat(vm_mae), stat(at_rel), stat(at_mae)
    return "\n".join([
        f"\nresults — {n_cases} cases ({cases_name}):",
        f"  {'V_m':<4}Rel L2  {vr[0]:.4f} ± {vr[1]:.4f}      MAE  {vmae[0]:.2f} ± {vmae[1]:.2f} mV",
        f"  {'AT':<4}Rel L2  {ar[0]:.4f} ± {ar[1]:.4f}      MAE  {amae[0]:.2f} ± {amae[1]:.2f} ms"])


# ════════════════════ V_m(t) traces ════════════════════
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


# ════════════════════ 3D scatter renders ════════════════════
# Run in a process pool; the shared xyz is passed once via the pool initializer
# (it is too large to re-pickle per task).
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


def render_snapshots(out_dir, xyz, pred, truth, at_pred, at_true, time_ms, labels):
    """3D V_m field snapshots (a few frames) + AT maps, one folder per case:
    snapshots/<case>/t###ms_*.svg and activation_time/<case>/AT_*.svg."""
    snap_root = os.path.join(out_dir, "snapshots")
    at_root = os.path.join(out_dir, "activation_time")
    targets = np.arange(0, min(300.0, time_ms[-1]) + 1, 10)       # ~10 ms steps
    frame_idx = [int(np.argmin(np.abs(time_ms - t))) for t in targets]

    tasks = []
    for c, label in enumerate(labels):
        snap_dir = os.path.join(snap_root, label); os.makedirs(snap_dir, exist_ok=True)
        at_dir = os.path.join(at_root, label); os.makedirs(at_dir, exist_ok=True)
        for fi in frame_idx:
            tag = f"t{time_ms[fi]:03.0f}ms"
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
            tasks.append((ap, CMAP_AT, lo, hi, os.path.join(at_dir, "AT_Pred.svg")))
            tasks.append((at, CMAP_AT, lo, hi, os.path.join(at_dir, "AT_GT.svg")))
            tasks.append((np.abs(at - ap), CMAP_ERR, 0.0, float(np.nanmax(np.abs(at - ap))),
                          os.path.join(at_dir, "AT_AbsErr.svg")))
        else:
            tasks.append((ap, CMAP_AT, float(np.nanmin(ap)), float(np.nanmax(ap)),
                          os.path.join(at_dir, "AT_Pred.svg")))

    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_plot_worker,
                             initargs=(xyz,)) as pool:
        list(pool.map(_render_scatter_svg, tasks))
    print(f"  rendered {len(tasks)} scatter SVGs -> {snap_root}/<case> , {at_root}/<case>")


# ════════════════════ standalone colorbars ════════════════════
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


def save_colorbars(out_dir, at_pred, at_true):
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


# ════════════════════ ParaView .vtu series ════════════════════
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
