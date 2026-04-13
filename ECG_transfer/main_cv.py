"""
ECG Transfer 5-fold cross-validation for all 3 model variants.

Shuffles the 125 cases with a fixed seed, splits into 5 folds of 25 each.
Each fold: 100 train (95 fit + 5 val for early stopping) / 25 test.
Reports per-fold and pooled Rel L2 and MAE on both raw 10-electrode
and derived 12-lead ECG.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/ECG_transfer

    python main_cv.py --model linear --epochs 5000 --device cuda
    python main_cv.py --model mlp --epochs 5000 --device cuda
    python main_cv.py --model temporal_conv --epochs 5000 --device cuda
"""
import os
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import LinearTransfer, MLPTransfer, TemporalConvTransfer

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"
N_FOLDS = 5

MODEL_MAP = {
    'linear': LinearTransfer,
    'mlp': MLPTransfer,
    'temporal_conv': TemporalConvTransfer,
}

ELECTRODE_NAMES = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_12lead(raw_10):
    """Convert (T, 10) electrode potentials to (T, 12) standard leads."""
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


def train_fold(fold_id, train_idx, val_idx, test_idx,
               vm_all, ecg_all, case_names,
               model_name, epochs, device, save_dir):
    """Train and evaluate one fold. Returns per-case metrics."""

    batch_size = 10
    learning_rate = 1e-4
    n_nodes = vm_all.shape[1]
    n_electrodes = ecg_all.shape[2]

    # Split
    vm_train = vm_all[train_idx]
    ecg_train = ecg_all[train_idx]
    vm_val = vm_all[val_idx]
    ecg_val = ecg_all[val_idx]
    vm_test = vm_all[test_idx]
    ecg_test = ecg_all[test_idx]

    # Normalization (per-fold, from training set only)
    vm_min = vm_train.min()
    vm_std = vm_train.std()
    vm_train_n = (vm_train - vm_min) / vm_std
    vm_val_n = (vm_val - vm_min) / vm_std
    vm_test_n = (vm_test - vm_min) / vm_std

    ecg_min = ecg_train.min()
    ecg_std = ecg_train.std()
    ecg_train_n = (ecg_train - ecg_min) / ecg_std
    ecg_val_n = (ecg_val - ecg_min) / ecg_std

    # Tensors
    vm_train_t = torch.tensor(vm_train_n, dtype=torch.float).to(device)
    ecg_train_t = torch.tensor(ecg_train_n, dtype=torch.float).to(device)
    vm_val_t = torch.tensor(vm_val_n, dtype=torch.float).to(device)
    ecg_val_t = torch.tensor(ecg_val_n, dtype=torch.float).to(device)
    vm_test_t = torch.tensor(vm_test_n, dtype=torch.float).to(device)

    train_ds = TensorDataset(vm_train_t, ecg_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MODEL_MAP[model_name](n_nodes=n_nodes, n_electrodes=n_electrodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        for b_vm, b_ecg in train_loader:
            optimizer.zero_grad()
            pred = model(b_vm)
            loss = ((pred - b_ecg) ** 2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = ((model(vm_val_t) - ecg_val_t) ** 2).mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)

        if epoch % 500 == 0:
            print(f"  Fold {fold_id} Epoch {epoch}: val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f}", flush=True)

    # Load best model and evaluate
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_norm = to_numpy(model(vm_test_t))
    # Denormalize
    pred_phy = pred_norm * ecg_std + ecg_min  # (num_test, T, 10)

    # Per-case metrics: raw 10-electrode
    l2_raw, mae_raw = [], []
    # Per-case metrics: derived 12-lead
    l2_12, mae_12 = [], []

    for i in range(len(test_idx)):
        p = pred_phy[i]       # (T, 10)
        t = ecg_test[i]       # (T, 10)
        l2 = np.linalg.norm(p - t) / np.linalg.norm(t)
        mae = np.mean(np.abs(p - t))
        l2_raw.append(l2)
        mae_raw.append(mae)

        p12 = to_12lead(p)
        t12 = to_12lead(t)
        l2 = np.linalg.norm(p12 - t12) / np.linalg.norm(t12)
        mae = np.mean(np.abs(p12 - t12))
        l2_12.append(l2)
        mae_12.append(mae)

    return l2_raw, mae_raw, l2_12, mae_12, pred_phy, ecg_test


def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description='ECG Transfer CV')
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'mlp', 'temporal_conv'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_name = args.model
    epochs = args.epochs
    device = args.device

    save_dir = f"CV_{N_FOLDS}fold_{model_name}_{epochs}ep"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = np.load(os.path.join(DATA_BASE, "geo_dynet_data.npz"),
                   allow_pickle=True)
    vm_all = data['vm'].astype(np.float32)       # (125, 50797, 121)
    ecg_all = data['ecg'].astype(np.float32)      # (125, 121, 10)
    time_ms = data['time'].astype(np.float32)     # (121,)
    case_names = data['case_names']

    n_total = len(vm_all)
    print(f"Data: {n_total} cases, {vm_all.shape[1]} nodes, "
          f"{vm_all.shape[2]} frames, {ecg_all.shape[2]} electrodes")
    print(f"Model: {model_name}, {N_FOLDS}-fold CV, {epochs} epochs/fold\n")

    # Shuffle indices (same seed as Geo-DeepONet CV for comparability)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    fold_size = n_total // N_FOLDS  # 25

    all_l2_raw, all_mae_raw = [], []
    all_l2_12, all_mae_12 = [], []
    all_case_results = []

    tic_total = time.time()
    for fold in range(N_FOLDS):
        print(f"=== Fold {fold} ===")
        tic = time.time()

        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        remain_idx = np.concatenate([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:]
        ])
        val_idx = remain_idx[-5:]
        train_idx = remain_idx[:-5]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, "
              f"Test: {len(test_idx)}")

        l2_raw, mae_raw, l2_12, mae_12, pred_phy, ecg_test = train_fold(
            fold, train_idx, val_idx, test_idx,
            vm_all, ecg_all, case_names,
            model_name, epochs, device, save_dir)

        all_l2_raw.extend(l2_raw)
        all_mae_raw.extend(mae_raw)
        all_l2_12.extend(l2_12)
        all_mae_12.extend(mae_12)

        for i, ti in enumerate(test_idx):
            all_case_results.append((
                case_names[ti], l2_raw[i], mae_raw[i], l2_12[i], mae_12[i]
            ))

        fold_time = int((time.time() - tic) / 60)
        print(f"  Fold {fold}: 10-elec Rel L2 = {np.mean(l2_raw):.4f} +/- "
              f"{np.std(l2_raw):.4f}, 12-lead Rel L2 = {np.mean(l2_12):.4f} "
              f"+/- {np.std(l2_12):.4f} ({fold_time} min)\n")

        # Plot 12-lead for first test case of each fold
        ti_first = test_idx[0]
        pred_12 = to_12lead(pred_phy[0])
        true_12 = to_12lead(ecg_test[0])
        layout = [
            ["I",   "aVR", "V1", "V4"],
            ["II",  "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        true_d = {n: true_12[:, j] for j, n in enumerate(LEAD_NAMES)}
        pred_d = {n: pred_12[:, j] for j, n in enumerate(LEAD_NAMES)}

        fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
        for row_idx, row in enumerate(layout):
            for col_idx, ln in enumerate(row):
                ax = axes[row_idx, col_idx]
                ax.plot(time_ms, true_d[ln] * 1000, 'k-', lw=0.8, label='True')
                ax.plot(time_ms, pred_d[ln] * 1000, 'r--', lw=0.8, label='Pred')
                ax.set_title(ln, fontsize=10, pad=2)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel('\u00b5V', fontsize=7)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)
        for ax in axes[3, :]:
            ax.set_visible(False)
        ax_rhythm = fig.add_subplot(4, 1, 4)
        ax_rhythm.plot(time_ms, true_d["II"] * 1000, 'k-', lw=0.8, label='True')
        ax_rhythm.plot(time_ms, pred_d["II"] * 1000, 'r--', lw=0.8, label='Pred')
        ax_rhythm.set_title('II (Rhythm Strip)', fontsize=10, pad=2)
        ax_rhythm.set_xlabel('Time (ms)')
        ax_rhythm.set_ylabel('\u00b5V', fontsize=7)
        ax_rhythm.grid(True, alpha=0.3)
        ax_rhythm.legend(fontsize=7)
        fig.suptitle(f'12-Lead ECG \u2014 {case_names[ti_first]} '
                     f'(Fold {fold}, {model_name})', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ecg_12lead_fold{fold}.png'), dpi=150)
        plt.close()

    total_time = int((time.time() - tic_total) / 60)

    # Summary
    print("=" * 80)
    print(f"{'Case':<35} {'10-elec L2':>12} {'10-elec MAE':>12} "
          f"{'12-lead L2':>12} {'12-lead MAE':>12}")
    print("-" * 80)
    for name, l2r, maer, l212, mae12 in sorted(all_case_results):
        print(f"{name:<35} {l2r:12.4f} {maer:12.6f} {l212:12.4f} {mae12:12.6f}")
    print("-" * 80)
    print(f"{'Pooled Mean':<35} {np.mean(all_l2_raw):12.4f} "
          f"{np.mean(all_mae_raw):12.6f} {np.mean(all_l2_12):12.4f} "
          f"{np.mean(all_mae_12):12.6f}")
    print(f"{'Pooled Std':<35} {np.std(all_l2_raw):12.4f} "
          f"{np.std(all_mae_raw):12.6f} {np.std(all_l2_12):12.4f} "
          f"{np.std(all_mae_12):12.6f}")
    print(f"\nTotal time: {total_time} min")

    # Save results
    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(results_path,
                        case_names=np.array([r[0] for r in all_case_results]),
                        l2_raw=np.array([r[1] for r in all_case_results]),
                        mae_raw=np.array([r[2] for r in all_case_results]),
                        l2_12lead=np.array([r[3] for r in all_case_results]),
                        mae_12lead=np.array([r[4] for r in all_case_results]),
                        fold_indices=indices)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
