"""
Visualize reconstruction error vs #temporal-PCA-components K.

Reads the small artefacts written by pca_temporal_analysis.py
(recon_vs_k.csv + the basis npz) — NO multi-GB data reload, runs in seconds.

The story this figure tells: cumulative *variance* is saturated by K~5-10
(the plateau), but the *application* errors that matter — V_m Rel L2, activation
-time MAE, and especially the upstroke steepness (max dV/dt) — keep falling out
to K~50-100. The current best NN is drawn as a horizontal reference so you can
read off the K at which a perfect-coefficient PCA decoder overtakes it.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# current best NN numbers (Geo_DONet, from VANDA_HANDOFF Results) for reference
NN_VM_RELL2 = 0.143      # SIREN+ndiff f121 best; Tanh baseline ~0.152
NN_AT_MAE = 5.13         # ms, SIREN+ndiff f301 best; Tanh ~6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/svu/e1032484/scratch/pca_temporal/recon_vs_k.csv")
    ap.add_argument("--basis", default="/home/svu/e1032484/scratch/pca_temporal_basis.npz")
    ap.add_argument("--out", default="/home/svu/e1032484/scratch/pca_temporal/error_vs_k.png")
    ap.add_argument("--tag", default="f601 (1 ms)")
    args = ap.parse_args()

    # K,cumEVR,vm_rel_l2,vm_mae,at_mae_ms,dvdt_med,dvdt_frac
    t = np.genfromtxt(args.csv, delimiter=",", names=True)
    K = t["K"]; cumEVR = t["cumEVR"]; rel_l2 = t["vm_rel_l2"]
    at_mae = t["at_mae_ms"]; dvdt_frac = t["dvdt_frac"]

    b = np.load(args.basis)
    full_K = np.arange(1, len(b["cum_evr"]) + 1)
    full_cum = b["cum_evr"]

    # K at which the perfect-coefficient PCA decoder first beats the NN
    def first_below(x, thr):
        hit = np.where(x <= thr)[0]
        return int(K[hit[0]]) if len(hit) else None
    k_beat_rel = first_below(rel_l2, NN_VM_RELL2)
    k_beat_at = first_below(at_mae, NN_AT_MAE)

    # variance milestones from the dense curve
    def k_for(frac):
        i = np.searchsorted(full_cum, frac)
        return int(full_K[i]) if i < len(full_K) else None
    kv = {f: k_for(f) for f in (0.90, 0.95, 0.99, 0.999)}

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    # ---- left: explained variance (what "how many components" naively means) ----
    a = ax[0]
    a.plot(full_K, full_cum, "-", color="C0", lw=2)
    a.scatter(K, cumEVR, s=18, color="C0", zorder=5)
    for f, c in zip((0.90, 0.95, 0.99, 0.999), ("grey",) * 4):
        a.axhline(f, color=c, ls=":", lw=0.8)
        a.annotate(f"{f*100:.1f}%  K={kv[f]}", (full_K[-1], f), fontsize=8,
                   ha="right", va="bottom", color="grey")
    a.set_xscale("log"); a.set_xlim(1, full_K[-1]); a.set_ylim(0.55, 1.005)
    a.set_xlabel("# components K (log)"); a.set_ylabel("cumulative explained variance")
    a.set_title("Variance view — saturates by K~10\n(misleading: plateau dominates)")
    a.grid(alpha=0.3, which="both")

    # ---- right: the errors that actually matter ----
    a = ax[1]
    l1, = a.semilogx(K, rel_l2, "o-", color="C0", label="V_m Rel L2")
    l3, = a.semilogx(K, 1 - dvdt_frac, "s-", color="C2",
                     label="upstroke deficit (1 - dV/dt frac)")
    a.axhline(NN_VM_RELL2, color="C0", ls="--", lw=1)
    a.annotate("current NN V_m Rel L2", (1.05, NN_VM_RELL2), color="C0",
               fontsize=8, va="bottom")
    if k_beat_rel:
        a.axvline(k_beat_rel, color="k", ls=":", lw=0.8)
        a.annotate(f"PCA beats NN\nat K={k_beat_rel}", (k_beat_rel, 0.18),
                   fontsize=8, ha="left")
    a.set_xlabel("# components K (log)"); a.set_ylabel("error (dimensionless, 0-1)")
    a.set_ylim(0, 0.22); a.grid(alpha=0.3, which="both")

    # twin axis: activation-time MAE in ms
    a2 = a.twinx()
    l2, = a2.semilogx(K, at_mae, "^-", color="C3", label="AT MAE (ms)")
    a2.axhline(NN_AT_MAE, color="C3", ls="--", lw=1)
    a2.annotate("current NN AT MAE", (K[-1], NN_AT_MAE), color="C3",
                fontsize=8, ha="right", va="bottom")
    a2.set_ylabel("activation-time MAE (ms)", color="C3")
    a2.tick_params(axis="y", labelcolor="C3"); a2.set_ylim(0, 12)

    a.legend(handles=[l1, l3, l2], loc="upper right", fontsize=9)
    a.set_title("Fidelity view — upstroke & AT need K~50-100\n"
                "(true-coefficient ceiling of the PCA-decoder plan)")

    fig.suptitle(f"Reconstruction error vs temporal-PCA components — {args.tag}, "
                 f"25 held-out test hearts", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")
    print(f"  variance milestones: {kv}")
    print(f"  PCA beats NN V_m Rel L2 ({NN_VM_RELL2}) at K={k_beat_rel}")
    print(f"  PCA beats NN AT MAE ({NN_AT_MAE} ms) at K={k_beat_at}")


if __name__ == "__main__":
    main()
