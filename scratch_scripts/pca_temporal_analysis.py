"""
Temporal-PCA analysis of nodewise V_m(t).
=========================================

Question 1 (variance): how many temporal PCA components are needed to capture
the nodewise membrane-potential waveform V_m(t)?

Question 2 (the one that matters for the NN plan): the smooth plateau dominates
variance, but the SHARP DEPOLARIZATION UPSTROKE is what every Tanh/SIREN/ndiff
run has failed to represent. So we report reconstruction fidelity vs K not just
as global Rel-L2, but specifically on the upstroke (max dV/dt) and the
activation time — the exact diagnostics the overfit capacity test used.

Method
------
Pool every (heart, node) pair into a sample matrix  X  of shape (C*N, T), each
row one node's V_m time series. Fit PCA (temporal eigen-waveforms, shared across
all nodes) on the TRAIN hearts only. Each node is then K coefficients; the
"inverse PCA" reconstruction is  mu + coeff[:, :K] @ V[:, :K].T .

We evaluate reconstruction on the HELD-OUT TEST hearts — that is the honest
number for the proposed pipeline (an operator net would predict these
coefficients for unseen geometries). The basis (mu, V, eigvals) is saved so the
NN track can consume it directly as a fixed decoder.

Split / units / AT definition all match Geo_DONet (95/5/25, mV, first upward
crossing of -10 mV). Imports the repo's own eval helpers so metrics are
identical to main.py --test-model.

Run on a compute node (cpu_parallel) — see pca_analysis.pbs. CPU only.
"""
import argparse
import os
import sys

import numpy as np

# use the repo's exact metric definitions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Geo_DONet"))
from utils import load_dataset, split_indices, vm_rel_l2_mae, activation_time  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def covariance_chunked(X, mu, chunk=200_000):
    """Accumulate (X-mu)^T (X-mu) in float64 without materialising a centred copy
    of the whole matrix. X is (n_samples, T)."""
    T = X.shape[1]
    cov = np.zeros((T, T), dtype=np.float64)
    n = X.shape[0]
    for s in range(0, n, chunk):
        block = X[s:s + chunk].astype(np.float64)
        block -= mu
        cov += block.T @ block
    return cov / (n - 1)


def max_dvdt(vm_2d, dt):
    """Per-node max dV/dt (mV/ms). vm_2d (n_nodes, T)."""
    return np.abs(np.gradient(vm_2d, dt, axis=1)).max(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/svu/e1032484/scratch/geo_donet_data_f601.npz")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="subsample frames (1=f601/1ms, 2=f301/2ms, 5=f121/5ms)")
    ap.add_argument("--n-train", type=int, default=95)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--at-threshold", type=float, default=-10.0)
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200])
    ap.add_argument("--out", default="/home/svu/e1032484/scratch/pca_temporal")
    ap.add_argument("--basis-out", default="/home/svu/e1032484/scratch/pca_temporal_basis.npz")
    ap.add_argument("--save-modes", type=int, default=200, help="how many modes to store in the basis npz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"=== temporal-PCA analysis ===")
    log(f"data         : {args.data}")
    d = load_dataset(args.data)
    vm = d["vm"]                 # (C, N, T) float32, mV
    time = d["time"]             # (T,) ms
    if args.frame_step > 1:
        vm = vm[:, :, ::args.frame_step]
        time = time[::args.frame_step]
    C, N, T = vm.shape
    dt = float(np.median(np.diff(time)))
    log(f"vm shape     : {vm.shape}  (C hearts, N nodes, T frames)")
    log(f"time         : {time[0]:.1f}..{time[-1]:.1f} ms, dt={dt:.3f} ms, frame_step={args.frame_step}")
    log(f"vm range     : [{vm.min():.2f}, {vm.max():.2f}] mV")

    train_idx, val_idx, test_idx = split_indices(C, args.n_train, args.n_val)
    log(f"split        : {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test (by heart)")

    # ---- fit basis on TRAIN node-waveforms ----
    Xtr = vm[train_idx].reshape(-1, T)          # (n_train*N, T) view->reshape
    mu = Xtr.mean(axis=0).astype(np.float64)
    log(f"fitting PCA on {Xtr.shape[0]:,} train node-waveforms (T={T}) ...")
    cov = covariance_chunked(Xtr, mu)
    eigval, eigvec = np.linalg.eigh(cov)        # ascending
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0, None)
    V = eigvec[:, order]                        # (T, T) columns = temporal modes, descending
    evr = eigval / eigval.sum()
    cum_evr = np.cumsum(evr)

    # K to reach variance milestones
    def k_for(frac):
        idx = np.searchsorted(cum_evr, frac)
        return int(idx + 1) if idx < T else T
    log("")
    log("--- global explained variance ---")
    for frac in (0.90, 0.95, 0.99, 0.999, 0.9999):
        log(f"  {frac*100:7.2f}% variance : K = {k_for(frac)}")

    # ---- reconstruction on TEST hearts vs K ----
    mu32 = mu.astype(np.float32)
    V32 = V.astype(np.float32)
    Xte = vm[test_idx].reshape(-1, T)
    coeff = (Xte.astype(np.float32) - mu32) @ V32          # (n_test*N, T)
    true_te = vm[test_idx]                                  # (n_test, N, T)
    n_te = len(test_idx)

    # per-node true upstroke + AT, computed once (full reconstruction == K=T baseline)
    true_dvdt_med = np.median([np.median(max_dvdt(true_te[c], dt)) for c in range(n_te)])

    log("")
    log("--- reconstruction on TEST hearts vs #components K ---")
    header = f"{'K':>5} {'cumEVR':>9} {'Vm RelL2':>9} {'Vm MAE':>8} {'AT MAE':>8} {'dVdt med':>9} {'dVdt frac':>9}"
    log(header)
    rows = []
    Kmax = min(max(args.k_list), T)
    for K in [k for k in args.k_list if k <= T]:
        recon = (mu32 + coeff[:, :K] @ V32[:, :K].T).reshape(n_te, N, T)
        rel_l2, mae = vm_rel_l2_mae(recon, true_te)
        # AT MAE over nodes activating in both
        at_maes = []
        for c in range(n_te):
            at_p = activation_time(recon[c], time, args.at_threshold)
            at_t = activation_time(true_te[c], time, args.at_threshold)
            valid = np.isfinite(at_p) & np.isfinite(at_t)
            if valid.any():
                at_maes.append(np.abs(at_p[valid] - at_t[valid]).mean())
        at_mae = float(np.mean(at_maes)) if at_maes else float("nan")
        rec_dvdt_med = np.median([np.median(max_dvdt(recon[c], dt)) for c in range(n_te)])
        frac = rec_dvdt_med / true_dvdt_med
        rows.append((K, cum_evr[K - 1], rel_l2.mean(), mae.mean(), at_mae, rec_dvdt_med, frac))
        log(f"{K:>5} {cum_evr[K-1]:>9.5f} {rel_l2.mean():>9.4f} {mae.mean():>8.3f} "
            f"{at_mae:>8.3f} {rec_dvdt_med:>9.2f} {frac:>9.3f}")
        del recon
    log(f"  (true median max dV/dt = {true_dvdt_med:.2f} mV/ms; dVdt frac = recon/true)")

    rows = np.array(rows, dtype=np.float64)

    # ---- save basis for the NN track ----
    nmodes = min(args.save_modes, T)
    np.savez(args.basis_out,
             mean_waveform=mu32, components=V32[:, :nmodes], eigval=eigval[:nmodes],
             evr=evr[:nmodes], cum_evr=cum_evr[:nmodes], time=time,
             at_threshold=np.float32(args.at_threshold), n_train=np.int64(args.n_train),
             frame_step=np.int64(args.frame_step))
    log(f"\nsaved basis -> {args.basis_out}  (mean + {nmodes} modes)")

    # ---- save table ----
    csv = os.path.join(args.out, "recon_vs_k.csv")
    np.savetxt(csv, rows, delimiter=",", header="K,cumEVR,vm_rel_l2,vm_mae,at_mae_ms,dvdt_med,dvdt_frac",
               comments="", fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])
    log(f"saved table -> {csv}")

    # ---- plots ----
    # 1. explained variance
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(np.arange(1, T + 1), evr, ".-")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("explained var ratio"); ax[0].set_title("per-mode EVR")
    ax[0].set_xlim(0, min(T, 200))
    ax[1].plot(np.arange(1, T + 1), cum_evr, ".-")
    for frac in (0.9, 0.99, 0.999):
        ax[1].axhline(frac, color="grey", ls=":", lw=0.8)
    ax[1].set_xlabel("# components K"); ax[1].set_ylabel("cumulative EVR"); ax[1].set_title("cumulative variance")
    ax[1].set_xlim(0, min(T, 200)); ax[1].set_ylim(0.5, 1.001)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "explained_variance.png"), dpi=120); plt.close(fig)

    # 2. reconstruction metrics vs K
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogx(rows[:, 0], rows[:, 2], "o-"); ax[0].set_title("test V_m Rel L2"); ax[0].set_xlabel("K")
    ax[1].semilogx(rows[:, 0], rows[:, 4], "o-"); ax[1].set_title("test AT MAE (ms)"); ax[1].set_xlabel("K")
    ax[2].semilogx(rows[:, 0], rows[:, 6], "o-"); ax[2].axhline(1.0, color="grey", ls=":")
    ax[2].set_title("upstroke fidelity: recon/true median max dV/dt"); ax[2].set_xlabel("K"); ax[2].set_ylim(0, 1.1)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "recon_vs_k.png"), dpi=120); plt.close(fig)

    # 3. first 6 temporal eigen-waveforms
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, mu32, "k-", lw=2, label="mean waveform")
    for k in range(6):
        ax.plot(time, V32[:, k] * np.sqrt(eigval[k]), label=f"mode {k+1} ({evr[k]*100:.1f}%)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("mV (mode scaled by sqrt eigval)")
    ax.set_title("mean + leading temporal modes"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "temporal_modes.png"), dpi=120); plt.close(fig)

    # 4. money plot: node-trace reconstructions at a few K, spanning early->late activation
    c0 = 0  # first test heart
    at_t0 = activation_time(true_te[c0], time, args.at_threshold)
    active = np.where(np.isfinite(at_t0))[0]
    order_at = active[np.argsort(at_t0[active])]
    picks = order_at[np.linspace(0, len(order_at) - 1, 4).astype(int)]
    K_show = [k for k in (5, 20, 50) if k <= T]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.6), sharey=True)
    for j, node in enumerate(picks):
        a = axes[j]
        a.plot(time, true_te[c0, node], "k-", lw=2, label="GT")
        full = (Xte.astype(np.float32) - mu32)[c0 * N + node]  # this node's centred series
        for K in K_show:
            rec = mu32 + (full @ V32[:, :K]) @ V32[:, :K].T
            a.plot(time, rec, lw=1, label=f"K={K}")
        a.set_title(f"node {node}, AT={at_t0[node]:.0f} ms", fontsize=9)
        a.set_xlabel("time (ms)"); a.grid(alpha=0.3)
        if j == 0:
            a.set_ylabel("V_m (mV)"); a.legend(fontsize=8)
    fig.suptitle(f"test heart {test_idx[c0]} — PCA reconstruction of V_m(t) vs K")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "node_traces.png"), dpi=120); plt.close(fig)

    log(f"\nplots -> {args.out}/{{explained_variance,recon_vs_k,temporal_modes,node_traces}}.png")
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("done.")


if __name__ == "__main__":
    main()
