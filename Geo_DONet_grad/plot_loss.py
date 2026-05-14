"""plot_loss.py — redraw the loss curve from saved train/val/test txt files.

Note: any test_loss.txt produced by the pre-2026-04-15 main.py actually holds
validation loss (mis-filed). Pass --treat-test-as-val to relabel it.

Usage:
    python plot_loss.py <run_dir>
    python plot_loss.py <run_dir> --treat-test-as-val
    python plot_loss.py <run_dir> --out custom.svg
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    return np.loadtxt(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='e.g. Predictions/geo_donet_5000ep_0.0005lr_w300')
    ap.add_argument('--treat-test-as-val', action='store_true',
                    help='label test_loss.txt as "Val" (for pre-2026-04-15 runs '
                         'where test_loss.txt was actually val loss)')
    ap.add_argument('--out', default=None,
                    help='output path (default <run_dir>/loss_curve_plot.svg)')
    a = ap.parse_args()

    train = load(os.path.join(a.run_dir, 'train_loss.txt'))
    val = load(os.path.join(a.run_dir, 'val_loss.txt'))
    test = load(os.path.join(a.run_dir, 'test_loss.txt'))

    if a.treat_test_as_val and test is not None:
        if val is None:
            val, test = test, None
            print("Relabelled test_loss.txt as Val (val_loss.txt not present).")
        else:
            print("Both val_loss.txt and test_loss.txt present; ignoring "
                  "--treat-test-as-val.")

    fig, ax = plt.subplots(figsize=(8, 5))
    if train is not None: ax.semilogy(train, label='Train', alpha=0.8)
    if val   is not None: ax.semilogy(val,   label='Val',   alpha=0.8)
    if test  is not None: ax.semilogy(test,  label='Test',  alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(os.path.basename(os.path.normpath(a.run_dir)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = a.out or os.path.join(a.run_dir, 'loss_curve_plot.svg')
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
