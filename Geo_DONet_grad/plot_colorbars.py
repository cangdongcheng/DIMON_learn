"""plot_colorbars.py — standalone, geometry-controlled colorbar SVGs.

Produces 4 SVGs with identical bar width and identical label/tick font sizes.
Vm and AT are rendered at full height; the two error bars at half height.
Geometry is set in absolute inches (not via bbox_inches='tight'), so the
bar columns line up pixel-for-pixel when scaled in a figure editor.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DONet

    # Option A — pull AT ranges from a saved test_predictions.npz
    python plot_colorbars.py \\
        --outdir Predictions/geo_donet_5000ep_w300/Test/colorbars \\
        --pred-npz Predictions/geo_donet_5000ep_w300/Test/test_predictions.npz

    # Option B — pass all ranges explicitly
    python plot_colorbars.py \\
        --outdir <outdir> \\
        --at-min 0 --at-max 160 --at-err-max 20
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ── Appearance knobs ─────────────────────────────────────────────────────
CMAP_VM = 'RdYlBu_r'
CMAP_AT = 'RdYlBu_r'
CMAP_ERR = 'Reds'

LABEL_FONTSIZE = 15
TICK_FONTSIZE = 15

# ── Geometry (inches). Same bar width for all four files. ────────────────
FIG_W_IN = 1.8          # figure width
BAR_LEFT_IN = 0.20      # left offset of bar column
BAR_W_IN = 0.40         # bar column width  (IDENTICAL for all 4 bars)
BAR_H_FULL_IN = 3.0     # Vm & AT
BAR_H_HALF_IN = 1.5     # |ΔV_m|, |ΔAT|
PAD_TOP_IN = 0.30
PAD_BOT_IN = 0.40


def make_bar(cmap, vmin, vmax, label, out_path, full=True):
    bar_h = BAR_H_FULL_IN if full else BAR_H_HALF_IN
    fig_h = bar_h + PAD_TOP_IN + PAD_BOT_IN
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([
        BAR_LEFT_IN / FIG_W_IN,
        PAD_BOT_IN / fig_h,
        BAR_W_IN / FIG_W_IN,
        bar_h / fig_h,
    ])
    ax.patch.set_alpha(0.0)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, cax=ax)

    cb.set_label(label, fontsize=LABEL_FONTSIZE)
    nice = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    nice = nice[(nice >= vmin) & (nice <= vmax)]
    cb.set_ticks(nice)
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)

    # Do NOT pass bbox_inches='tight' — it would re-crop per-file and break
    # the pixel alignment we just set up.
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close()


def ranges_from_npz(path):
    """Return (at_vmin, at_vmax, at_err_vmax) from a saved test_predictions.npz."""
    d = np.load(path, allow_pickle=True)
    at_pred = d['at_pred']       # (num_test, M)
    at_true = d['at_true']       # (num_test, M)
    # Match what main.py saves for the viz hearts (heart 0 and 1)
    viz = slice(0, min(2, at_pred.shape[0]))
    at_stack = np.concatenate([at_pred[viz].ravel(), at_true[viz].ravel()])
    at_vmin = float(np.nanmin(at_stack))
    at_vmax = float(np.nanmax(at_stack))
    err = np.abs(at_pred[viz] - at_true[viz])
    at_err_vmax = float(np.nanmax(err))
    return at_vmin, at_vmax, at_err_vmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', required=True)
    p.add_argument('--pred-npz', default=None,
                   help='Read AT ranges from this test_predictions.npz '
                        '(overrides --at-min/--at-max/--at-err-max).')
    p.add_argument('--vm-min', type=float, default=-90.0)
    p.add_argument('--vm-max', type=float, default=50.0)
    p.add_argument('--vm-err-min', type=float, default=0.0)
    p.add_argument('--vm-err-max', type=float, default=20.0)
    p.add_argument('--at-min', type=float, default=0.0)
    p.add_argument('--at-max', type=float, default=160.0)
    p.add_argument('--at-err-min', type=float, default=0.0)
    p.add_argument('--at-err-max', type=float, default=20.0)
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    if a.pred_npz:
        at_vmin, at_vmax, at_err_vmax = ranges_from_npz(a.pred_npz)
        print(f"AT ranges from {a.pred_npz}: "
              f"[{at_vmin:.2f}, {at_vmax:.2f}], err_max={at_err_vmax:.2f}")
    else:
        at_vmin, at_vmax, at_err_vmax = a.at_min, a.at_max, a.at_err_max

    make_bar(CMAP_VM, a.vm_min, a.vm_max, 'V_m (mV)',
             os.path.join(a.outdir, 'cbar_Vm.svg'), full=True)
    make_bar(CMAP_AT, at_vmin, at_vmax, 'AT (ms)',
             os.path.join(a.outdir, 'cbar_AT.svg'), full=True)
    make_bar(CMAP_ERR, a.vm_err_min, a.vm_err_max, '|ΔV_m| (mV)',
             os.path.join(a.outdir, 'cbar_Vm_AbsErr.svg'), full=False)
    make_bar(CMAP_ERR, a.at_err_min, at_err_vmax, '|ΔAT| (ms)',
             os.path.join(a.outdir, 'cbar_AT_AbsErr.svg'), full=False)

    print(f"Saved 4 colorbars to {a.outdir}")


if __name__ == '__main__':
    main()
