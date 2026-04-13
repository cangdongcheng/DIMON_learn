"""
Simple bar plot: per-fold Rel L2 and MAE with individual data points.
Style matches the DIMON paper's tenfold CV figure.

Usage:
    python plot_cv_bars.py
    python plot_cv_bars.py --cv-dir CV_5fold_3000ep
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
    l2_all = data["l2_errors"]
    mae_all = data["mae_errors"]
    fold_indices = data["fold_indices"]

    # Reconstruct per-fold grouping (25 cases per fold)
    n_folds = 5
    fold_size = 25
    fold_l2 = [l2_all[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    fold_mae = [mae_all[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]

    fold_labels = [f"Fold {i}" for i in range(n_folds)]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_folds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # --- Rel L2 ---
    means_l2 = [np.mean(f) for f in fold_l2]
    stds_l2 = [np.std(f) for f in fold_l2]
    x = np.arange(n_folds)

    ax1.bar(x, means_l2, yerr=stds_l2, width=0.6, color=colors, alpha=0.7,
            edgecolor="black", linewidth=0.5, capsize=4, error_kw={"linewidth": 1})
    for i in range(n_folds):
        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(fold_l2[i]))
        ax1.scatter(x[i] + jitter, fold_l2[i], s=12, color=colors[i],
                    edgecolor="black", linewidth=0.3, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontsize=9)
    ax1.set_ylabel("Relative L$_2$ error")
    ax1.set_title("Relative L$_2$ error")

    # --- MAE ---
    means_mae = [np.mean(f) for f in fold_mae]
    stds_mae = [np.std(f) for f in fold_mae]

    ax2.bar(x, means_mae, yerr=stds_mae, width=0.6, color=colors, alpha=0.7,
            edgecolor="black", linewidth=0.5, capsize=4, error_kw={"linewidth": 1})
    for i in range(n_folds):
        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(fold_mae[i]))
        ax2.scatter(x[i] + jitter, fold_mae[i], s=12, color=colors[i],
                    edgecolor="black", linewidth=0.3, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels, fontsize=9)
    ax2.set_ylabel("MAE (ms)")
    ax2.set_title("Absolute error (ms)")

    fig.suptitle("5-fold cross-validation", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(args.cv_dir, "cv_bars.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
