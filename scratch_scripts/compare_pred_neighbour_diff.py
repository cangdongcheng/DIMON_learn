"""
Predicted vs ground-truth Vm neighbour differences, on the mesh + as curves.

Consumes a model's test_predictions.npz (keys: pred, true, case_names) -- already
nodewise, denormalised mV, canonical node order -- so NO inference is rerun.
For one test heart it runs both pred and true through the SAME neighbour-diff
computation as the GT script and compares them.

The headline question (does an MSE-trained model reproduce the sharp wavefront,
or smooth it -- i.e. is a gradient loss warranted?) is answered by the overlaid
spatial-gradient signal  sum_edges (V_d - V_s)^2  vs time.

OUTPUTS (under --out-dir/pred_<case>/):
    gradient_signal_gt_vs_pred.png   the two energy curves overlaid
    neighbour_diff_loss.png          candidate matching losses vs t:
        mean_node (ndiff_pred-ndiff_true)^2  and  mean_node (signed_pred-signed_true)^2
    comparison_summary.vtu           static mesh, per-node:
        vm_mae            mean_t |pred-true|        (where the model errs)
        ndiff_max_true / ndiff_max_pred             (sharpness, each)
        signed_atpeak_true / signed_atpeak_pred     (source/sink at the GT peak frame)
        ndiff_err_mae / signed_err_mae              (where a matching loss would bite)
    series/frame_####.vtu + comparison.pvd
        per frame: vm_true, vm_pred, vm_err, ndiff_true/pred/err,
        signed_true/pred/err   (err = pred - true == the candidate loss field)
    comparison_fields.npz            + ndiff_err, signed_err, per-frame loss curves,
                                       and scalar Vm/ndiff/signed MSEs

RUN:
    source ~/load_dimon_env.sh
    cd ~/DIMON_learn
    python scratch_scripts/compare_pred_neighbour_diff.py --heart 100
    # --heart is the GLOBAL index 0..124 (same convention as the GT script);
    # predictions exist only for the 25 TEST hearts = global 100..124.
    # Or select unambiguously with --case IMMC201_20170214_merge_000
"""
import argparse
import os
import time as timer

import numpy as np
import meshio

# reuse the exact edge build + field reduction from the GT script (same dir)
from visualize_neighbour_diff import build_edges, compute_fields

SCRATCH = os.environ.get("DIMON_DATA_BASE", "/home/svu/e1032484/scratch")
DEFAULT_PRED = ("/home/svu/e1032484/DIMON_learn/Geo_DONet/Predictions/"
                "geo_donet_5000ep_w300_lrsched/Test/test_predictions.npz")


def write_static(path, points, tets, ndt, nds, sgt, sgs, vm_true, vm_pred, peak):
    """ndt/nds: ndiff true/pred (N,T); sgt/sgs: signed true/pred (N,T)."""
    meshio.write_points_cells(
        path, points, [("tetra", tets)],
        point_data={
            "vm_mae": np.abs(vm_pred - vm_true).mean(axis=1).astype(np.float32),
            "ndiff_max_true": ndt.max(axis=1).astype(np.float32),
            "ndiff_max_pred": nds.max(axis=1).astype(np.float32),
            "signed_atpeak_true": sgt[:, peak].astype(np.float32),
            "signed_atpeak_pred": sgs[:, peak].astype(np.float32),
            # mean-over-time |pred - true| of each field == where a matching loss bites
            "ndiff_err_mae": np.abs(nds - ndt).mean(axis=1).astype(np.float32),
            "signed_err_mae": np.abs(sgs - sgt).mean(axis=1).astype(np.float32),
        })


def write_series(out_dir, points, tets, time_ms, fields, stride):
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
    with open(os.path.join(out_dir, "comparison.pvd"), "w") as f:
        f.write("\n".join(pvd))
    return len(sel)


def plot_overlay(path, e_true, e_pred, time_ms, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_ms, e_true, lw=1.8, color="k", label="ground truth")
    ax.plot(time_ms, e_pred, lw=1.8, color="r", label="prediction")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$\sum_e (V_d - V_s)^2$  [mV$^2$]")
    ax.set_title(f"{case}: spatial-gradient signal  GT vs Pred")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_loss_curves(path, ndiff_err, signed_err, time_ms, case):
    """Per-frame mean-over-nodes squared error of each field = candidate loss vs t."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nl = (ndiff_err ** 2).mean(axis=0)
    sl = (signed_err ** 2).mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time_ms, nl, lw=1.6, color="C0",
            label="unsigned: mean (ndiff_pred - ndiff_true)^2")
    ax.plot(time_ms, sl, lw=1.6, color="C3",
            label="signed: mean (signed_pred - signed_true)^2")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean over nodes (pred - true)^2   [mV^2]")
    ax.set_title(f"{case}: candidate neighbour-diff matching loss vs t")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-file", default=DEFAULT_PRED)
    ap.add_argument("--heart", type=int, default=100,
                    help="GLOBAL case index 0..124 (same as the GT script); "
                         "predictions exist only for test hearts 100..124")
    ap.add_argument("--case", default=None, help="select by case name instead of index")
    ap.add_argument("--vtu", default=os.path.join(SCRATCH, "canonical.vtu"))
    ap.add_argument("--data-file", default="geo_donet_data_f121.npz",
                    help="source of the time axis (must match pred frame count)")
    ap.add_argument("--out-dir", default=os.path.join(SCRATCH, "neighbour_diff"))
    ap.add_argument("--vtu-stride", type=int, default=0, help="0=auto ~60 frames; 1=all")
    ap.add_argument("--no-series", action="store_true")
    args = ap.parse_args()

    t0 = timer.time()
    m = meshio.read(args.vtu)
    points = m.points.astype(np.float32)
    tets = next(cb.data for cb in m.cells if cb.type == "tetra").astype(np.int32)
    N = points.shape[0]
    esrc, edst = build_edges(tets, N)

    pz = np.load(args.pred_file, allow_pickle=True)
    test_names = [str(x) for x in pz["case_names"]]

    data = np.load(os.path.join(SCRATCH, args.data_file), mmap_mode="r")
    all_names = [str(x) for x in data["case_names"]]
    time_ms = np.asarray(data["time"], dtype=np.float32)
    test_start = len(all_names) - len(test_names)        # = 100 (95 train + 5 val)

    # Resolve to a GLOBAL case index (0..124), same convention as the GT script.
    if args.case:
        if args.case not in all_names:
            raise SystemExit(f"unknown case {args.case!r}")
        gidx = all_names.index(args.case)
    else:
        gidx = args.heart
    tpos = gidx - test_start
    if not (0 <= tpos < len(test_names)):
        nm = all_names[gidx] if 0 <= gidx < len(all_names) else "?"
        raise SystemExit(
            f"heart {gidx} ({nm}) has no saved prediction. test_predictions.npz "
            f"covers GLOBAL indices {test_start}..{test_start + len(test_names) - 1} "
            f"(the {len(test_names)} test hearts) only. Pick one of those "
            f"(e.g. --heart {test_start}) or run inference for a non-test heart.")
    assert test_names[tpos] == all_names[gidx], "split/index mismatch"
    case = all_names[gidx]
    pred = np.ascontiguousarray(pz["pred"][tpos]).astype(np.float32)   # (N,T)
    true = np.ascontiguousarray(pz["true"][tpos]).astype(np.float32)   # (N,T)
    assert pred.shape[0] == N, "pred/mesh node mismatch"
    T = pred.shape[1]
    assert time_ms.shape[0] == T, f"time has {time_ms.shape[0]} frames, pred has {T}"

    print(f"heart {gidx} (test pos {tpos}) = {case}: "
          f"Vm true[{true.min():.1f},{true.max():.1f}] "
          f"pred[{pred.min():.1f},{pred.max():.1f}] mV", flush=True)

    ndt, sgt, et = compute_fields(true, esrc, edst, N)
    nds, sgs, ep = compute_fields(pred, esrc, edst, N)
    peak = int(np.argmax(et))                       # compare signed at the GT peak frame
    print(f"peak spatial-gradient signal @ t={time_ms[peak]:.0f}ms: "
          f"true={et[peak]:.3e}  pred={ep[peak]:.3e}  "
          f"pred/true={ep[peak]/et[peak]:.2f}", flush=True)

    # pred - true error of each neighbour-diff field == the candidate matching loss
    ndiff_err = nds - ndt
    signed_err = sgs - sgt
    vm_mse = float(((pred - true) ** 2).mean())
    ndiff_mse = float((ndiff_err ** 2).mean())
    signed_mse = float((signed_err ** 2).mean())
    print(f"candidate losses (mean over nodes & frames):  Vm MSE={vm_mse:.4f}  |  "
          f"unsigned-ndiff match MSE={ndiff_mse:.4f}  |  "
          f"signed match MSE={signed_mse:.4f}   [mV^2]", flush=True)

    out_dir = os.path.join(args.out_dir, f"pred_{case}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "comparison_fields.npz"),
                        ndiff_true=ndt, ndiff_pred=nds, signed_true=sgt,
                        signed_pred=sgs, ndiff_err=ndiff_err, signed_err=signed_err,
                        energy_true=et, energy_pred=ep,
                        ndiff_loss_t=(ndiff_err ** 2).mean(axis=0),
                        signed_loss_t=(signed_err ** 2).mean(axis=0),
                        vm_mse=vm_mse, ndiff_mse=ndiff_mse, signed_mse=signed_mse,
                        time=time_ms, case=case)

    plot_overlay(os.path.join(out_dir, "gradient_signal_gt_vs_pred.png"),
                 et, ep, time_ms, case)
    plot_loss_curves(os.path.join(out_dir, "neighbour_diff_loss.png"),
                     ndiff_err, signed_err, time_ms, case)
    write_static(os.path.join(out_dir, "comparison_summary.vtu"),
                 points, tets, ndt, nds, sgt, sgs, true, pred, peak)
    print(f"plots+static -> {out_dir}", flush=True)

    if not args.no_series:
        stride = args.vtu_stride or max(1, round(T / 60))
        n = write_series(out_dir, points, tets, time_ms, {
            "vm_true": true, "vm_pred": pred, "vm_err": np.abs(pred - true),
            "ndiff_true": ndt, "ndiff_pred": nds, "ndiff_err": ndiff_err,
            "signed_true": sgt, "signed_pred": sgs, "signed_err": signed_err}, stride)
        print(f"series -> {out_dir}/comparison.pvd ({n} frames, stride {stride})",
              flush=True)

    print(f"\nDONE in {timer.time()-t0:.1f}s -> {out_dir}\n"
          f"Start with gradient_signal_gt_vs_pred.png (does pred match the GT peak?).")


if __name__ == "__main__":
    main()