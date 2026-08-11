"""
Fourier (FFT) truncation of nodewise V_m(t) — K-vs-error, compared to PCA.
==========================================================================

Same experiment as pca_temporal_analysis.py but with a FIXED Fourier basis
instead of the data-adapted PCA basis. For each node we rfft its V_m(t), keep
the lowest `m` frequencies (low-pass), irfft back, and measure the same errors
(V_m Rel L2, activation-time MAE, upstroke max-dV/dt fraction) on the same 25
held-out test hearts.

Fairness: a Fourier frequency is a COMPLEX coefficient = 2 real numbers, so
keeping m frequencies costs (2m-1) real DOF (DC is real). PCA mode = 1 real
coeff. We therefore plot everything against *real DOF* so the curves overlay
honestly — that is also "how many numbers the NN must predict per node".

Because PCA is the optimal MSE basis, the Fourier curve should sit at or ABOVE
(worse than) the PCA curve. How close they are tells you whether a fixed,
fit-free Fourier feature set is good enough vs needing the data-fit PCA basis.

A second, dashed "adaptive best-m-term" Fourier curve (keep each node's m
LARGEST coefficients) is an oracle upper bound — NOT a usable fixed feature set
(different frequencies per node), shown only to bound the gap.

Run on the interactive node (CPU). Reuses Geo_DONet's exact metric helpers.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Geo_DONet"))
from utils import load_dataset, split_indices, vm_rel_l2_mae, activation_time  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def max_dvdt(vm_2d, dt):
    return np.abs(np.gradient(vm_2d, dt, axis=1)).max(axis=1)


def metrics_for_recon(recon, true_te, time, dt, at_threshold, true_dvdt_med):
    """recon/true_te: (n_te, N, T). Returns (vm_rel_l2, vm_mae, at_mae, dvdt_med, dvdt_frac)."""
    n_te = recon.shape[0]
    rel_l2, mae = vm_rel_l2_mae(recon, true_te)
    at_maes = []
    for c in range(n_te):
        at_p = activation_time(recon[c], time, at_threshold)
        at_t = activation_time(true_te[c], time, at_threshold)
        valid = np.isfinite(at_p) & np.isfinite(at_t)
        if valid.any():
            at_maes.append(np.abs(at_p[valid] - at_t[valid]).mean())
    at_mae = float(np.mean(at_maes)) if at_maes else float("nan")
    rec_dvdt_med = np.median([np.median(max_dvdt(recon[c], dt)) for c in range(n_te)])
    return rel_l2.mean(), mae.mean(), at_mae, rec_dvdt_med, rec_dvdt_med / true_dvdt_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/svu/e1032484/scratch/geo_donet_data_f601.npz")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=95)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--at-threshold", type=float, default=-10.0)
    ap.add_argument("--m-list", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 11, 15, 20, 25, 38, 50, 75, 100],
                    help="number of lowest complex frequencies to keep")
    ap.add_argument("--adaptive", action="store_true",
                    help="also compute the oracle best-m-term (largest-coeff) curve")
    ap.add_argument("--pca-csv", default="/home/svu/e1032484/scratch/pca_temporal/recon_vs_k.csv")
    ap.add_argument("--out", default="/home/svu/e1032484/scratch/fourier_trunc")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    d = load_dataset(args.data)
    vm = d["vm"]; time = d["time"]
    if args.frame_step > 1:
        vm = vm[:, :, ::args.frame_step]
        time = time[::args.frame_step]
    C, N, T = vm.shape
    dt = float(np.median(np.diff(time)))
    n_freq = T // 2 + 1
    log(f"=== Fourier truncation analysis ===")
    log(f"vm {vm.shape}  dt={dt:.3f} ms  rfft -> {n_freq} complex freqs (max real DOF {2*n_freq-1 if T%2 else 2*n_freq-2})")

    _, _, test_idx = split_indices(C, args.n_train, args.n_val)
    true_te = vm[test_idx]                                  # (n_te, N, T)
    n_te = len(test_idx)
    X = true_te.reshape(-1, T).astype(np.float32)           # (n_te*N, T)
    log(f"test hearts: {len(test_idx)} -> {X.shape[0]:,} node-waveforms")

    true_dvdt_med = np.median([np.median(max_dvdt(true_te[c], dt)) for c in range(n_te)])
    log(f"true median max dV/dt = {true_dvdt_med:.2f} mV/ms")

    F = np.fft.rfft(X, axis=1)                              # (n, n_freq) complex
    mag = np.abs(F)

    def real_dof(m):
        # DC is real; for odd T no Nyquist-real term, so each of the other (m-1) freqs is 2 real DOF
        return 2 * m - 1

    log("")
    log("--- low-pass Fourier (keep lowest m freqs) ---")
    log(f"{'m':>5} {'realDOF':>8} {'Vm RelL2':>9} {'Vm MAE':>8} {'AT MAE':>8} {'dVdt med':>9} {'dVdt frac':>9}")
    rows = []
    for m in [m for m in args.m_list if m <= n_freq]:
        Fk = np.zeros_like(F)
        Fk[:, :m] = F[:, :m]
        recon = np.fft.irfft(Fk, n=T, axis=1).astype(np.float32).reshape(n_te, N, T)
        rl, ma, at, dm, df = metrics_for_recon(recon, true_te, time, dt, args.at_threshold, true_dvdt_med)
        rows.append((m, real_dof(m), rl, ma, at, dm, df))
        log(f"{m:>5} {real_dof(m):>8} {rl:>9.4f} {ma:>8.3f} {at:>8.3f} {dm:>9.2f} {df:>9.3f}")
        del recon, Fk
    rows = np.array(rows, dtype=np.float64)
    np.savetxt(os.path.join(args.out, "fourier_lowpass_vs_m.csv"), rows, delimiter=",",
               header="m,real_dof,vm_rel_l2,vm_mae,at_mae_ms,dvdt_med,dvdt_frac",
               comments="", fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])

    adapt = None
    if args.adaptive:
        log("")
        log("--- oracle adaptive best-m-term (per-node largest |coeff|; NOT a fixed feature set) ---")
        a_rows = []
        # rank freqs per node by magnitude once
        order = np.argsort(-mag, axis=1)                   # (n, n_freq)
        for m in [m for m in args.m_list if m <= n_freq]:
            keep = order[:, :m]
            Fk = np.zeros_like(F)
            np.put_along_axis(Fk, keep, np.take_along_axis(F, keep, axis=1), axis=1)
            recon = np.fft.irfft(Fk, n=T, axis=1).astype(np.float32).reshape(n_te, N, T)
            rl, ma, at, dm, df = metrics_for_recon(recon, true_te, time, dt, args.at_threshold, true_dvdt_med)
            a_rows.append((m, real_dof(m), rl, ma, at, dm, df))
            log(f"{m:>5} {real_dof(m):>8} {rl:>9.4f} {ma:>8.3f} {at:>8.3f} {dm:>9.2f} {df:>9.3f}")
            del recon, Fk
        adapt = np.array(a_rows, dtype=np.float64)

    # ---- overlay figure vs PCA ----
    pca = None
    if os.path.exists(args.pca_csv):
        pca = np.genfromtxt(args.pca_csv, delimiter=",", names=True)  # K == real DOF for PCA

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [(2, "V_m Rel L2", None), (4, "activation-time MAE (ms)", None),
             (6, "upstroke fidelity (recon/true dV/dt)", (0, 1.1))]
    pca_cols = {2: "vm_rel_l2", 4: "at_mae_ms", 6: "dvdt_frac"}
    for j, (col, title, ylim) in enumerate(specs):
        a = ax[j]
        if pca is not None:
            a.semilogx(pca["K"], pca[pca_cols[col]], "o-", color="C0", label="PCA (data-optimal)")
        a.semilogx(rows[:, 1], rows[:, col], "s-", color="C3", label="Fourier low-pass")
        if adapt is not None:
            a.semilogx(adapt[:, 1], adapt[:, col], "^--", color="C1", lw=1,
                       label="Fourier best-m-term (oracle)")
        if col == 6:
            a.axhline(1.0, color="grey", ls=":")
        a.set_xlabel("real coefficients per node"); a.set_title(title)
        a.grid(alpha=0.3, which="both")
        if ylim:
            a.set_ylim(*ylim)
        if j == 0:
            a.legend(fontsize=9)
    fig.suptitle("Fixed Fourier basis vs data-optimal PCA — V_m(t) reconstruction, 25 test hearts",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(args.out, "fourier_vs_pca.png")
    fig.savefig(out_png, dpi=130)
    log(f"\nsaved -> {out_png}")
    log(f"saved table -> {args.out}/fourier_lowpass_vs_m.csv")
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("done.")


if __name__ == "__main__":
    main()
