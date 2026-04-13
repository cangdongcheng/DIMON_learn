"""
eval_chain.py — Two-stage inference: Geo-DONet V_m → ECG_transfer.

Evaluates three pipelines on the 25 held-out test cases:
  A. ECG_transfer(GT V_m)         — upper bound: only transfer error
  B. ECG_transfer(Geo-DONet V_m)  — chain: adds V_m prediction error
  C. Geo-DONet-ECG (joint)        — optional, pass --joint-ckpt

Key insight: both models use identical V_m normalisation
  (vm_mean = vm_train.min(), vm_std = vm_train.std())
so Geo-DONet's normalised output feeds directly into ECG_transfer
without any intermediate conversion.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON
    python eval_chain.py \\
        --geodnet-ckpt Geo_DONet/CheckPts/model_chkpts_geo_donet_5000ep_0.0005lr_w300.pt \\
        --ecg-ckpt     ECG_transfer/CheckPts/temporal_conv_10000ep.pt \\
        --device cuda

    # Once Geo-DONet-ECG job finishes, add:
        --joint-ckpt Geo_DONet_ECG/CheckPts/model_geo_donet_ecg_5000ep_0.0005lr_lambda1.0.pt
"""
import argparse
import os
import sys

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- import from subdirs using importlib to avoid opnn.py name collision ---
# Three subdirs each have their own opnn.py; sys.path shadowing would break things.
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))


def _load(module_name, rel_path):
    path = os.path.join(BASE, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_geo_opnn   = _load('geo_opnn',   'Geo_DONet/opnn.py')
_ecg_model  = _load('ecg_model',  'ECG_transfer/model.py')
_joint_opnn = _load('joint_opnn', 'Geo_DONet_ECG/opnn.py')

GeoDONet             = _geo_opnn.opnn
TemporalConvTransfer = _ecg_model.TemporalConvTransfer
GeoDONetECG          = _joint_opnn.GeoDONetECG

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"


# ---------------------------------------------------------------------------
# Trunk helpers (mirror of Geo_DONet/main.py — kept local to avoid import
# coupling across subdirs)
# ---------------------------------------------------------------------------

def precompute_trunk_chunks(coords, time_points, chunk_size, device):
    """Build (M*T_c, 5) trunk input grids for all temporal chunks."""
    M = coords.shape[0]
    T = len(time_points)
    chunks = []
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        T_c = t_end - t_start
        coords_rep = coords.unsqueeze(1).expand(M, T_c, 4).reshape(M * T_c, 4)
        time_rep = time_points[t_start:t_end].unsqueeze(0).expand(M, T_c).reshape(M * T_c, 1)
        chunks.append(torch.cat([coords_rep, time_rep], dim=1))
    return chunks


def chunked_forward(model, f_geo, trunk_chunks, M):
    """
    Run Geo-DONet with precomputed trunk chunks.
    f_geo: (N, 60), trunk_chunks: list of (M*T_c, 5)
    Returns: (N, M, T) normalised V_m.
    """
    y_br = model._branch_g(f_geo)                  # (N, p)
    outputs = []
    for xt_chunk in trunk_chunks:
        T_c = xt_chunk.shape[0] // M
        y_tr = model.encode_trunk(xt_chunk)         # (M*T_c, p)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr)  # (N, M*T_c)
        outputs.append(y_out.reshape(f_geo.shape[0], M, T_c))
    return torch.cat(outputs, dim=2)               # (N, M, T)


# ---------------------------------------------------------------------------
# 12-lead conversion (mirror of ECG_transfer/main.py)
# Electrode order: LA, RA, LL, RL, V1–V6
# ---------------------------------------------------------------------------

def to_12lead(raw_10):
    """(T, 10) electrode potentials → (T, 12) standard leads."""
    LA, RA, LL = raw_10[:, 0], raw_10[:, 1], raw_10[:, 2]
    WCT = (RA + LA + LL) / 3
    return np.column_stack([
        LA - RA,                # I
        LL - RA,                # II
        LL - LA,                # III
        RA - (LA + LL) / 2,     # aVR
        LA - (RA + LL) / 2,     # aVL
        LL - (RA + LA) / 2,     # aVF
        raw_10[:, 4] - WCT,     # V1
        raw_10[:, 5] - WCT,     # V2
        raw_10[:, 6] - WCT,     # V3
        raw_10[:, 7] - WCT,     # V4
        raw_10[:, 8] - WCT,     # V5
        raw_10[:, 9] - WCT,     # V6
    ])


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def per_case_metrics(preds_phy, gt_phy, case_names, label):
    """Print per-case 10-elec and 12-lead metrics, return arrays."""
    n = len(preds_phy)
    l2_10, l2_12 = [], []
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'Case':<35} {'10-elec Rel L2':>14} {'12-lead Rel L2':>14}")
    print(f"  {'─'*63}")
    for i in range(n):
        r10 = rel_l2(preds_phy[i], gt_phy[i])
        p12 = to_12lead(preds_phy[i])
        t12 = to_12lead(gt_phy[i])
        r12 = rel_l2(p12, t12)
        l2_10.append(r10)
        l2_12.append(r12)
        print(f"  {case_names[i]:<35} {r10:14.4f} {r12:14.4f}")
    print(f"  {'─'*63}")
    print(f"  {'Mean':<35} {np.mean(l2_10):14.4f} {np.mean(l2_12):14.4f}")
    print(f"  {'Std':<35} {np.std(l2_10):14.4f} {np.std(l2_12):14.4f}")
    return np.array(l2_10), np.array(l2_12)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ecg_comparison(time_ms, ecg_gt, pipelines, case_name, outdir):
    """
    pipelines: list of (label, ecg_phy (T, 10)) tuples.
    Saves 3×4 12-lead grid + rhythm strip for one test case.
    """
    true_12 = to_12lead(ecg_gt)
    true_dict = {n: true_12[:, j] for j, n in enumerate(LEAD_NAMES)}

    colors = ['C0', 'C1', 'C2', 'C3']

    fig, axes = plt.subplots(4, 4, figsize=(18, 10), sharex=True)
    for row_idx, row in enumerate(LAYOUT):
        for col_idx, ln in enumerate(row):
            ax = axes[row_idx, col_idx]
            ax.plot(time_ms, true_dict[ln] * 1000, 'k-', lw=1.2, label='GT')
            for ci, (lbl, ecg_phy) in enumerate(pipelines):
                pred_12 = to_12lead(ecg_phy)
                pred_dict = {n: pred_12[:, j] for j, n in enumerate(LEAD_NAMES)}
                ax.plot(time_ms, pred_dict[ln] * 1000,
                        color=colors[ci], lw=0.9, ls='--', label=lbl)
            ax.set_title(ln, fontsize=10, pad=2)
            ax.set_ylabel('µV', fontsize=7)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    # Rhythm strip (Lead II, full row)
    for ax in axes[3, :]:
        ax.set_visible(False)
    ax_rhythm = fig.add_subplot(4, 1, 4)
    ax_rhythm.plot(time_ms, true_dict["II"] * 1000, 'k-', lw=1.2, label='GT')
    for ci, (lbl, ecg_phy) in enumerate(pipelines):
        pred_12 = to_12lead(ecg_phy)
        ax_rhythm.plot(time_ms, pred_12[:, 1] * 1000,
                       color=colors[ci], lw=0.9, ls='--', label=lbl)
    ax_rhythm.set_title('II (Rhythm Strip)', fontsize=10, pad=2)
    ax_rhythm.set_xlabel('Time (ms)')
    ax_rhythm.set_ylabel('µV', fontsize=7)
    ax_rhythm.grid(True, alpha=0.3)
    ax_rhythm.legend(fontsize=7)

    fig.suptitle(f'12-Lead ECG — {case_name}', fontsize=12)
    plt.tight_layout()
    fname = os.path.join(outdir, f'ecg_{case_name}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_summary_bars(results, outdir):
    """
    results: list of (label, l2_10 array, l2_12 array).
    Side-by-side mean bars with std error bars.
    """
    labels = [r[0] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    mean10 = [r[1].mean() for r in results]
    std10  = [r[1].std()  for r in results]
    mean12 = [r[2].mean() for r in results]
    std12  = [r[2].std()  for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width/2, mean10, width, yerr=std10,
                label='10-elec Rel L2', capsize=4, alpha=0.8)
    b2 = ax.bar(x + width/2, mean12, width, yerr=std12,
                label='12-lead Rel L2', capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Relative L2 error')
    ax.set_title('ECG prediction — pipeline comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fname = os.path.join(outdir, 'pipeline_comparison.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Geo-DONet → ECG_transfer chain eval')
    parser.add_argument('--geodnet-ckpt',
                        default='Geo_DONet/CheckPts/model_chkpts_geo_donet_5000ep_0.0005lr_w300.pt')
    parser.add_argument('--ecg-ckpt',
                        default='ECG_transfer/CheckPts/temporal_conv_10000ep.pt')
    parser.add_argument('--joint-ckpt', default=None,
                        help='Geo-DONet-ECG checkpoint for optional pipeline C')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--outdir', default='chain_eval')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.outdir, exist_ok=True)
    sync = (lambda: torch.cuda.synchronize()) if 'cuda' in str(device) else (lambda: None)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    data = np.load(os.path.join(DATA_BASE, 'geo_dynet_data.npz'), allow_pickle=True)
    theta   = data['theta'][:, :60].astype(np.float32)   # (125, 60)
    coords  = data['coords'].astype(np.float32)           # (50797, 4)
    vm_all  = data['vm'].astype(np.float32)               # (125, 50797, 121)
    ecg_all = data['ecg'].astype(np.float32)              # (125, 121, 10)
    time_ms = data['time'].astype(np.float32)             # (121,)
    case_names = data['case_names']

    n_pts, n_time = vm_all.shape[1], vm_all.shape[2]
    num_train, num_val = 95, 5
    num_test = len(theta) - num_train - num_val
    test_names = case_names[num_train + num_val:]

    print(f"  {len(theta)} cases, {n_pts} nodes, {n_time} frames")
    print(f"  Split: {num_train} train / {num_val} val / {num_test} test")

    # ── Normalisation stats — training split only ──────────────────────────
    vm_train  = vm_all[:num_train]
    ecg_train = ecg_all[:num_train]
    theta_train = theta[:num_train]

    # Geometry z-score
    f_mean = theta_train.mean(axis=0)
    f_std  = theta_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0

    # V_m: min-shift + std-scale  (identical in Geo-DONet and ECG_transfer)
    vm_mean = vm_train.min()
    vm_std  = vm_train.std()

    # ECG: min-shift + std-scale
    ecg_mean = ecg_train.min()
    ecg_std  = ecg_train.std()

    # ── Test arrays ────────────────────────────────────────────────────────
    vm_test_raw  = vm_all[num_train + num_val:]    # (25, 50797, 121) physical mV
    ecg_test_raw = ecg_all[num_train + num_val:]   # (25, 121, 10) physical mV

    # Normalised inputs
    vm_test_norm = (vm_test_raw - vm_mean) / vm_std
    vm_test_t = torch.tensor(vm_test_norm, dtype=torch.float).to(device)

    theta_test_norm = (theta[num_train + num_val:] - f_mean) / f_std
    f_test_t = torch.tensor(theta_test_norm, dtype=torch.float).to(device)

    # ── Trunk setup for Geo-DONet ──────────────────────────────────────────
    coords_t  = torch.tensor(coords, dtype=torch.float).to(device)
    x_min = coords_t.min(dim=0, keepdim=True)[0]
    x_max = coords_t.max(dim=0, keepdim=True)[0]
    coords_norm = (coords_t - x_min) / (x_max - x_min + 1e-8)

    t_min, t_max = float(time_ms.min()), float(time_ms.max())
    time_norm = (time_ms - t_min) / (t_max - t_min + 1e-8)
    time_t = torch.tensor(time_norm, dtype=torch.float).to(device)

    trunk_chunks = precompute_trunk_chunks(coords_norm, time_t, 25, device)
    print(f"  Trunk: {len(trunk_chunks)} chunks")

    # ── Pipeline A: ECG_transfer on GT V_m (upper bound) ──────────────────
    print(f"\nLoading ECG_transfer: {args.ecg_ckpt}")
    ecg_model = TemporalConvTransfer(n_nodes=n_pts, n_electrodes=10).to(device)
    ckpt = torch.load(args.ecg_ckpt, map_location=device, weights_only=True)
    ecg_model.load_state_dict(ckpt['model_state_dict'])
    ecg_model.eval()

    print("  Running pipeline A: ECG_transfer(GT V_m)...")
    with torch.no_grad():
        ecg_A_norm = ecg_model(vm_test_t).cpu().numpy()   # (25, 121, 10)
    ecg_A = ecg_A_norm * ecg_std + ecg_mean               # physical mV

    # ── Geo-DONet: predict V_m (normalised) ───────────────────────────────
    print(f"\nLoading Geo-DONet: {args.geodnet_ckpt}")
    geo_model = GeoDONet(
        branch_geo_dim=[60, 300, 300, 300, 300],
        trunk_dim=[5, 300, 300, 300, 300],
    ).to(device)
    ckpt = torch.load(args.geodnet_ckpt, map_location=device, weights_only=True)
    geo_model.load_state_dict(ckpt['model_state_dict'])
    geo_model.eval()

    print("  Running Geo-DONet inference...")
    geo_times = []
    with torch.no_grad():
        preds = []
        for i in range(num_test):
            sync()
            t0 = time.perf_counter()
            p = chunked_forward(geo_model, f_test_t[i:i+1], trunk_chunks, n_pts)
            sync()
            geo_times.append(time.perf_counter() - t0)
            preds.append(p)
        vm_pred_norm_t = torch.cat(preds, dim=0)           # (25, 50797, 121)

    vm_pred_phy = vm_pred_norm_t.cpu().numpy() * vm_std + vm_mean

    # V_m quality (for reference)
    vm_l2 = [rel_l2(vm_pred_phy[i], vm_test_raw[i]) for i in range(num_test)]
    print(f"  Geo-DONet V_m Rel L2: {np.mean(vm_l2):.4f} ± {np.std(vm_l2):.4f}")

    # ── Pipeline B: ECG_transfer on Geo-DONet V_m ─────────────────────────
    # vm_pred_norm_t is already in the normalised space ECG_transfer expects —
    # both models use the same vm_mean/vm_std, so no conversion needed.
    print("  Running pipeline B: ECG_transfer(Geo-DONet V_m)...")
    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        ecg_B_norm = ecg_model(vm_pred_norm_t).cpu().numpy()  # (25, 121, 10)
        sync()
        ecg_B_time = time.perf_counter() - t0
    ecg_B = ecg_B_norm * ecg_std + ecg_mean

    # ── Pipeline C: Geo-DONet-ECG joint (optional) ─────────────────────────
    ecg_C = None
    if args.joint_ckpt:
        print(f"\nLoading Geo-DONet-ECG: {args.joint_ckpt}")
        joint_model = GeoDONetECG(
            branch_geo_dim=[60, 300, 300, 300, 300],
            trunk_dim=[5, 300, 300, 300, 300],
            n_nodes=n_pts, n_electrodes=10, ecg_mode='linear',
        ).to(device)
        ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=True)
        joint_model.load_state_dict(ckpt['model_state_dict'])
        joint_model.eval()

        print("  Running pipeline C: Geo-DONet-ECG...")
        with torch.no_grad():
            preds_c = []
            for i in range(num_test):
                vm_i = chunked_forward(joint_model, f_test_t[i:i+1], trunk_chunks, n_pts)
                ecg_i = joint_model.predict_ecg(vm_i)   # (1, 121, 10) normalised
                preds_c.append(ecg_i.cpu().numpy())
        ecg_C_norm = np.concatenate(preds_c, axis=0)   # (25, 121, 10)
        ecg_C = ecg_C_norm * ecg_std + ecg_mean

    # ── Metrics ────────────────────────────────────────────────────────────
    results = []
    l2A_10, l2A_12 = per_case_metrics(ecg_A, ecg_test_raw, test_names,
                                       'A: ECG_transfer(GT V_m) — upper bound')
    results.append(('A: Transfer\n(GT V_m)', l2A_10, l2A_12))

    l2B_10, l2B_12 = per_case_metrics(ecg_B, ecg_test_raw, test_names,
                                       'B: ECG_transfer(Geo-DONet V_m) — chain')
    results.append(('B: Transfer\n(GeoDONet)', l2B_10, l2B_12))

    if ecg_C is not None:
        l2C_10, l2C_12 = per_case_metrics(ecg_C, ecg_test_raw, test_names,
                                           'C: Geo-DONet-ECG — joint model')
        results.append(('C: Joint\nGeoDONet-ECG', l2C_10, l2C_12))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  SUMMARY (mean ± std, {num_test} test cases)")
    print(f"{'═'*50}")
    print(f"  {'Pipeline':<28} {'10-elec':>10} {'12-lead':>10}")
    print(f"  {'─'*48}")
    for lbl, r10, r12 in results:
        lbl_short = lbl.replace('\n', ' ')
        print(f"  {lbl_short:<28} {np.mean(r10):6.4f}±{np.std(r10):.4f}"
              f"  {np.mean(r12):6.4f}±{np.std(r12):.4f}")
    print(f"\n  V_m error (Geo-DONet): {np.mean(vm_l2):.4f} ± {np.std(vm_l2):.4f}")

    geo_total = sum(geo_times)
    chain_B_total = geo_total + ecg_B_time
    print(f"\n  Inference timing ({num_test} test cases)")
    print(f"  {'─'*56}")
    print(f"  {'Geo-DONet (total)':<30} {geo_total*1000:>8.1f} ms"
          f"  ({np.mean(geo_times)*1000:.1f} ± {np.std(geo_times)*1000:.1f} ms/case)")
    print(f"  {'ECG_transfer B (batched)':<30} {ecg_B_time*1000:>8.1f} ms"
          f"  ({ecg_B_time/num_test*1000:.1f} ms/case avg)")
    print(f"  {'Chain B total':<30} {chain_B_total*1000:>8.1f} ms"
          f"  ({chain_B_total/num_test*1000:.1f} ms/case avg)")

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\nSaving plots to {args.outdir}/")
    num_viz = min(3, num_test)
    for h in range(num_viz):
        pipelines = [
            ('A: Transfer(GT)', ecg_A[h]),
            ('B: Transfer(Chain)', ecg_B[h]),
        ]
        if ecg_C is not None:
            pipelines.append(('C: Joint', ecg_C[h]))
        plot_ecg_comparison(time_ms, ecg_test_raw[h], pipelines,
                            test_names[h], args.outdir)

    plot_summary_bars(results, args.outdir)

    # ── Save results ───────────────────────────────────────────────────────
    save_dict = dict(
        ecg_gt=ecg_test_raw, ecg_A=ecg_A, ecg_B=ecg_B,
        vm_pred=vm_pred_phy, vm_gt=vm_test_raw,
        l2_10_A=l2A_10, l2_12_A=l2A_12,
        l2_10_B=l2B_10, l2_12_B=l2B_12,
        vm_l2=np.array(vm_l2),
        case_names=test_names, time_ms=time_ms,
    )
    if ecg_C is not None:
        save_dict.update(ecg_C=ecg_C, l2_10_C=l2C_10, l2_12_C=l2C_12)

    np.savez_compressed(os.path.join(args.outdir, 'chain_results.npz'), **save_dict)
    print(f"  Saved {args.outdir}/chain_results.npz")
    print("Done.")


if __name__ == '__main__':
    main()
