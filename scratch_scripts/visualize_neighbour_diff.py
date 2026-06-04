"""
Ground-truth Vm neighbour differences, written onto the mesh as VTU + a
spatial-gradient-signal curve vs time.

For one heart, per node, per frame, from the Vm differences across each node's
mesh edges, compute:
    ndiff   = mean |Vm[i] - Vm[j]| over neighbours j      (unsigned magnitude)
    signed  = Vm[i] - mean_j Vm[j]                          (signed: + if the
              node is hotter than its neighbour average, - if cooler)
and the global spatial-gradient signal  energy(t) = sum_edges (V_d - V_s)^2.

Mesh + ordering provenance (verified exact, zero matching):
    canonical.vtu point order == reference_cobiveco.npz == geo_donet vm/coords,
so edges from canonical.vtu tets index directly into vm[node].

OUTPUTS (under --out-dir/<case>/), all plain .vtu / .png:
    neighbour_diff_summary.vtu   ONE static mesh, per-node fields:
        ndiff_max       max over time of the unsigned neighbour diff
        peak_time_ms    time of that max (activation-map-like)
        signed_at_peak  signed diff at the global peak-gradient frame
                        (source/sink pattern of the wavefront)
    series/frame_####.vtu + neighbour_diff.pvd
        time series, per-node: vm, ndiff, signed. Open the .pvd, colour by
        "signed" or "ndiff", press play.  (subsample with --vtu-stride)
    gradient_signal.png          sum_edges (V_d - V_s)^2  vs time  [mV^2]
    neighbour_diff_fields.npz    ndiff, signed, energy, time (for reuse)

RUN (CPU is fine, no torch):
    source ~/load_dimon_env.sh
    cd ~/DIMON_learn
    python scratch_scripts/visualize_neighbour_diff.py --heart 0
    # high-res 1 ms:   --data-file geo_donet_data_f601.npz
    # every frame:     --vtu-stride 1
"""
import argparse
import os
import time

import numpy as np
import meshio

SCRATCH = os.environ.get("DIMON_DATA_BASE", "/home/svu/e1032484/scratch")


def build_edges(tets, n_nodes):
    """Unique undirected edges from tet connectivity, via a 1-D hash."""
    pairs = tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]]
    e = pairs.reshape(-1, 2)
    e.sort(axis=1)
    key = np.unique(e[:, 0].astype(np.int64) * n_nodes + e[:, 1])
    return (key // n_nodes).astype(np.int64), (key % n_nodes).astype(np.int64)


def compute_fields(vm, esrc, edst, N):
    """One pass over frames. Returns:
       ndiff  (N,T) mean |dVm| over neighbours,
       signed (N,T) Vm - mean(neighbour Vm),
       energy (T,)  sum_edges (V_d - V_s)^2.
    Reduces frame-by-frame so the (E,T) edge array is never materialised."""
    T = vm.shape[1]
    deg = np.bincount(esrc, minlength=N) + np.bincount(edst, minlength=N)
    degf = np.maximum(deg, 1).astype(np.float64)
    ndiff = np.zeros((N, T), np.float64)
    signed = np.zeros((N, T), np.float64)
    energy = np.zeros(T, np.float64)
    for t in range(T):
        vs = vm[esrc, t].astype(np.float64)
        vd = vm[edst, t].astype(np.float64)
        diff = vd - vs
        energy[t] = float(diff @ diff)                      # sum_edges (V_d-V_s)^2
        ad = np.abs(diff)
        ndiff[:, t] = np.bincount(esrc, ad, N) + np.bincount(edst, ad, N)
        nbr_sum = np.bincount(esrc, vd, N) + np.bincount(edst, vs, N)
        signed[:, t] = vm[:, t].astype(np.float64) - nbr_sum / degf
    ndiff /= degf[:, None]
    return ndiff.astype(np.float32), signed.astype(np.float32), energy


def write_static_vtu(path, points, tets, ndiff, signed, energy, time_ms):
    peak = int(np.argmax(energy))
    meshio.write_points_cells(
        path, points, [("tetra", tets)],
        point_data={
            "ndiff_max": ndiff.max(axis=1).astype(np.float32),
            "peak_time_ms": time_ms[np.argmax(ndiff, axis=1)].astype(np.float32),
            "signed_at_peak": signed[:, peak].astype(np.float32),
        })


def write_vtu_series(out_dir, points, tets, time_ms, fields, stride):
    fdir = os.path.join(out_dir, "series")
    os.makedirs(fdir, exist_ok=True)
    cells = [("tetra", tets)]
    sel = list(range(0, time_ms.shape[0], max(1, stride)))
    rows = []
    for t in sel:
        fn = f"frame_{t:04d}.vtu"
        meshio.write_points_cells(os.path.join(fdir, fn), points, cells,
                                  point_data={k: v[:, t] for k, v in fields.items()})
        rows.append(f'    <DataSet timestep="{float(time_ms[t])}" group="" '
                    f'part="0" file="series/{fn}"/>')
    pvd = (['<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>'] + rows + ['  </Collection>', '</VTKFile>'])
    with open(os.path.join(out_dir, "neighbour_diff.pvd"), "w") as f:
        f.write("\n".join(pvd))
    return len(sel)


def plot_gradient_signal(path, energy, time_ms, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_ms, energy, lw=1.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$\sum_e (V_d - V_s)^2$  [mV$^2$]")
    ax.set_title(f"{case} - spatial-gradient signal vs t")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heart", type=int, default=0, help="case index (default 0)")
    ap.add_argument("--data-file", default="geo_donet_data_f121.npz")
    ap.add_argument("--vtu", default=os.path.join(SCRATCH, "canonical.vtu"))
    ap.add_argument("--out-dir", default=os.path.join(SCRATCH, "neighbour_diff"))
    ap.add_argument("--vtu-stride", type=int, default=0,
                    help="time-series frame stride (0 = auto ~60 frames; 1 = all)")
    ap.add_argument("--no-series", action="store_true", help="skip the .vtu time series")
    args = ap.parse_args()

    t0 = time.time()
    print(f"reading mesh {args.vtu}", flush=True)
    m = meshio.read(args.vtu)
    points = m.points.astype(np.float32)
    tets = next(cb.data for cb in m.cells if cb.type == "tetra").astype(np.int32)
    N = points.shape[0]
    esrc, edst = build_edges(tets, N)
    print(f"  N={N} tets={tets.shape[0]} edges={esrc.shape[0]}  ({time.time()-t0:.1f}s)")

    data = np.load(os.path.join(SCRATCH, args.data_file), mmap_mode="r")
    assert N == data["coords"].shape[0], "mesh/data node count mismatch"
    case = str(data["case_names"][args.heart])
    vm = np.ascontiguousarray(data["vm"][args.heart]).astype(np.float32)
    time_ms = np.asarray(data["time"], dtype=np.float32)
    print(f"heart {args.heart} = {case}: vm {vm.shape}  Vm[{vm.min():.1f},{vm.max():.1f}] mV")

    t1 = time.time()
    ndiff, signed, energy = compute_fields(vm, esrc, edst, N)
    print(f"fields in {time.time()-t1:.1f}s; ndiff max={ndiff.max():.2f} mV, "
          f"signed [{signed.min():.2f},{signed.max():.2f}] mV, "
          f"peak energy={energy.max():.3e} mV^2 at t={time_ms[int(np.argmax(energy))]:.0f}ms",
          flush=True)

    out_dir = os.path.join(args.out_dir, f"heart{args.heart}_{case}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "neighbour_diff_fields.npz"),
                        ndiff=ndiff, signed=signed, energy=energy, time=time_ms, case=case)

    write_static_vtu(os.path.join(out_dir, "neighbour_diff_summary.vtu"),
                     points, tets, ndiff, signed, energy, time_ms)
    print(f"static  -> {out_dir}/neighbour_diff_summary.vtu", flush=True)

    plot_gradient_signal(os.path.join(out_dir, "gradient_signal.png"),
                         energy, time_ms, case)
    print(f"plot    -> {out_dir}/gradient_signal.png", flush=True)

    if not args.no_series:
        stride = args.vtu_stride or max(1, round(time_ms.shape[0] / 60))
        n = write_vtu_series(out_dir, points, tets, time_ms,
                             {"vm": vm, "ndiff": ndiff, "signed": signed}, stride)
        print(f"series  -> {out_dir}/neighbour_diff.pvd  ({n} frames, stride {stride})",
              flush=True)

    print(f"\nDONE in {time.time()-t0:.1f}s -> {out_dir}\n"
          f"ParaView: open neighbour_diff.pvd, colour by 'signed' (diverging) or 'ndiff'.")


if __name__ == "__main__":
    main()