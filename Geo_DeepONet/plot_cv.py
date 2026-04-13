"""
Plot 5-fold CV results: per-case MAE and Rel L2 bar charts + distributions.

Usage:
    python plot_cv.py
    python plot_cv.py --cv-dir CV_5fold_3000ep
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", default="CV_5fold_3000ep")
    args = parser.parse_args()

    data = np.load(os.path.join(args.cv_dir, "cv_results.npz"),
                   allow_pickle=True)
    case_names = data["case_names"]
    l2_errors = data["l2_errors"]
    mae_errors = data["mae_errors"]

    n = len(case_names)
    # Short labels: IMMC001_20150716 → IMMC001
    short_names = [c.split("_")[0] for c in case_names]

    # Sort by MAE for the bar chart
    sort_idx = np.argsort(mae_errors)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Top left: MAE bar chart (sorted) ---
    ax = axes[0, 0]
    ax.barh(range(n), mae_errors[sort_idx], color="steelblue", edgecolor="none")
    ax.set_yticks(range(0, n, 5))
    ax.set_yticklabels([short_names[sort_idx[i]] for i in range(0, n, 5)],
                       fontsize=6)
    ax.set_xlabel("MAE (ms)")
    ax.set_title(f"Per-case MAE (sorted) — mean={np.mean(mae_errors):.2f} ms")
    ax.axvline(np.mean(mae_errors), color="red", linestyle="--", linewidth=1)
    ax.invert_yaxis()

    # --- Top right: Rel L2 bar chart (sorted) ---
    sort_idx_l2 = np.argsort(l2_errors)
    ax = axes[0, 1]
    ax.barh(range(n), l2_errors[sort_idx_l2], color="darkorange", edgecolor="none")
    ax.set_yticks(range(0, n, 5))
    ax.set_yticklabels([short_names[sort_idx_l2[i]] for i in range(0, n, 5)],
                       fontsize=6)
    ax.set_xlabel("Relative L2 Error")
    ax.set_title(f"Per-case Rel L2 (sorted) — mean={np.mean(l2_errors):.4f}")
    ax.axvline(np.mean(l2_errors), color="red", linestyle="--", linewidth=1)
    ax.invert_yaxis()

    # --- Bottom left: MAE histogram ---
    ax = axes[1, 0]
    ax.hist(mae_errors, bins=20, color="steelblue", edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(mae_errors), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(mae_errors):.2f}")
    ax.axvline(np.median(mae_errors), color="green", linestyle="--", linewidth=1.5,
               label=f"median={np.median(mae_errors):.2f}")
    ax.set_xlabel("MAE (ms)")
    ax.set_ylabel("Count")
    ax.set_title("MAE Distribution")
    ax.legend(fontsize=9)

    # --- Bottom right: Rel L2 histogram ---
    ax = axes[1, 1]
    ax.hist(l2_errors, bins=20, color="darkorange", edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(l2_errors), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(l2_errors):.4f}")
    ax.axvline(np.median(l2_errors), color="green", linestyle="--", linewidth=1.5,
               label=f"median={np.median(l2_errors):.4f}")
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Count")
    ax.set_title("Rel L2 Distribution")
    ax.legend(fontsize=9)

    fig.suptitle(f"Geo-DeepONet 5-Fold CV ({args.cv_dir}) — {n} cases",
                 fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(args.cv_dir, "cv_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
