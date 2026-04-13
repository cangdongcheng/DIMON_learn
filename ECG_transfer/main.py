"""
ECG Transfer (Approach B): learn V_m(x,t) → ECG(10,t) mapping.

Uses ground-truth V_m from monodomain simulations as input.
Tests whether the spatial-to-electrode mapping is learnable
before integrating with Geo-DONet.

Three model variants: linear, mlp, temporal_conv.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/ECG_transfer

    python main.py --model linear --epochs 5000 --device cuda
    python main.py --model mlp --epochs 5000 --device cuda
    python main.py --model temporal_conv --epochs 5000 --device cuda

    # Evaluate
    python main.py --model linear --epochs 5000 --test-model 1 --device cuda
"""
import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from utils import *
from model import LinearTransfer, MLPTransfer, TemporalConvTransfer

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"

MODEL_MAP = {
    'linear': LinearTransfer,
    'mlp': MLPTransfer,
    'temporal_conv': TemporalConvTransfer,
}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    import argparse
    parser = argparse.ArgumentParser(description='ECG Transfer')
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'mlp', 'temporal_conv'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test-model', type=int, default=0)
    parser.add_argument('--save-step', type=int, default=1000)
    args = parser.parse_args()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    model_name = args.model

    batch_size = 10
    learning_rate = 1e-4

    save_dir = f'Predictions/{model_name}_{epochs}ep'
    model_path = f'CheckPts/{model_name}_{epochs}ep.pt'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## Load data
    data = np.load(os.path.join(DATA_BASE, "geo_dynet_data.npz"),
                   allow_pickle=True)
    vm_all = data['vm'].astype(np.float32)       # (125, 50797, 121)
    ecg_all = data['ecg'].astype(np.float32)      # (125, 121, 10)
    time_ms = data['time'].astype(np.float32)     # (121,)
    case_names = data['case_names']

    n_total, n_nodes, n_time = vm_all.shape
    n_electrodes = ecg_all.shape[2]
    print(f"Data: {n_total} cases, {n_nodes} nodes, {n_time} frames, {n_electrodes} electrodes")

    ## Split
    num_train, num_val = 95, 5
    num_test = n_total - num_train - num_val

    vm_train = vm_all[:num_train]
    ecg_train = ecg_all[:num_train]
    vm_val = vm_all[num_train:num_train + num_val]
    ecg_val = ecg_all[num_train:num_train + num_val]
    vm_test = vm_all[num_train + num_val:]
    ecg_test = ecg_all[num_train + num_val:]

    ## Normalization
    # V_m: min-shift + std-scale
    vm_mean = vm_train.min()
    vm_std = vm_train.std()
    vm_train_n = (vm_train - vm_mean) / vm_std
    vm_val_n = (vm_val - vm_mean) / vm_std
    vm_test_n = (vm_test - vm_mean) / vm_std

    # ECG: same strategy
    ecg_mean = ecg_train.min()
    ecg_std = ecg_train.std()
    ecg_train_n = (ecg_train - ecg_mean) / ecg_std
    ecg_val_n = (ecg_val - ecg_mean) / ecg_std
    ecg_test_n = (ecg_test - ecg_mean) / ecg_std

    ## Tensors — all on GPU
    vm_train_t = torch.tensor(vm_train_n, dtype=torch.float).to(device)
    ecg_train_t = torch.tensor(ecg_train_n, dtype=torch.float).to(device)
    vm_val_t = torch.tensor(vm_val_n, dtype=torch.float).to(device)
    ecg_val_t = torch.tensor(ecg_val_n, dtype=torch.float).to(device)
    vm_test_t = torch.tensor(vm_test_n, dtype=torch.float).to(device)

    print(f"Split: {num_train} train / {num_val} val / {num_test} test")

    ## Model
    model = MODEL_MAP[model_name](n_nodes=n_nodes, n_electrodes=n_electrodes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if test_model == 0:
        ## Training
        train_ds = TensorDataset(vm_train_t, ecg_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        train_loss_his, val_loss_his = [], []
        best_val_loss = float('inf')

        print(f"--- Training {model_name} ---", flush=True)
        tic = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for b_vm, b_ecg in train_loader:
                optimizer.zero_grad()
                pred = model(b_vm)  # (B, T, 10)
                loss = ((pred - b_ecg) ** 2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_loss_his.append(avg_loss)

            model.eval()
            with torch.no_grad():
                pred_val = model(vm_val_t)
                val_loss = ((pred_val - ecg_val_t) ** 2).mean().item()
                val_loss_his.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if epoch % 100 == 0:
                elapsed = time.time() - tic
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                print(f'Epoch {epoch}/{epochs} | Train: {avg_loss:.6f} | '
                      f'Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | '
                      f'ETA: {int(eta//60)}m{int(eta%60)}s', flush=True)

        print(f"Training done: {int((time.time() - tic) / 60)} min")
        np.savetxt(os.path.join(save_dir, 'train_loss.txt'), train_loss_his)
        np.savetxt(os.path.join(save_dir, 'val_loss.txt'), val_loss_his)

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss_his, label='Train', alpha=0.8)
        ax.semilogy(val_loss_his, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{model_name}_{epochs}ep')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        print(f"Saved loss curve to {save_dir}/loss_curve.png")

    else:
        ## Evaluation
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"--- Evaluating {model_name} ---")

        with torch.no_grad():
            pred_test = to_numpy(model(vm_test_t))
        # Denormalize
        pred_phy = pred_test * ecg_std + ecg_mean  # (num_test, T, 10)

        # Per-case metrics
        electrode_names = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]
        print(f"\n{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>10}")
        print("-" * 58)
        l2_errors, mae_errors = [], []
        for i in range(num_test):
            p = pred_phy[i]       # (T, 10)
            t = ecg_test[i]       # (T, 10)
            l2 = np.linalg.norm(p - t) / np.linalg.norm(t)
            mae = np.mean(np.abs(p - t))
            l2_errors.append(l2)
            mae_errors.append(mae)
            print(f"{case_names[num_train + num_val + i]:<35}"
                  f" {l2:10.4f} {mae:10.6f}")
        print("-" * 58)
        print(f"{'Mean':<35} {np.mean(l2_errors):10.4f} {np.mean(mae_errors):10.6f}")
        print(f"{'Std':<35} {np.std(l2_errors):10.4f} {np.std(mae_errors):10.6f}")

        # --- Convert 10 electrodes → 12-lead ECG ---
        # Electrode order: LA, RA, LL, RL, V1, V2, V3, V4, V5, V6
        def to_12lead(raw_10):
            """Convert (T, 10) electrode potentials to (T, 12) standard leads."""
            LA, RA, LL = raw_10[:, 0], raw_10[:, 1], raw_10[:, 2]
            WCT = (RA + LA + LL) / 3
            leads = np.column_stack([
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
            return leads

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]

        # Per-case 12-lead metrics
        print(f"\n--- 12-Lead ECG Metrics ---")
        print(f"{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>10}")
        print("-" * 58)
        l2_12lead, mae_12lead = [], []
        for i in range(num_test):
            pred_12 = to_12lead(pred_phy[i])
            true_12 = to_12lead(ecg_test[i])
            l2 = np.linalg.norm(pred_12 - true_12) / np.linalg.norm(true_12)
            mae = np.mean(np.abs(pred_12 - true_12))
            l2_12lead.append(l2)
            mae_12lead.append(mae)
            print(f"{case_names[num_train + num_val + i]:<35}"
                  f" {l2:10.4f} {mae:10.6f}")
        print("-" * 58)
        print(f"{'Mean':<35} {np.mean(l2_12lead):10.4f} {np.mean(mae_12lead):10.6f}")
        print(f"{'Std':<35} {np.std(l2_12lead):10.4f} {np.std(mae_12lead):10.6f}")

        # --- Plot: 10-electrode raw traces ---
        num_viz = min(2, num_test)
        for h in range(num_viz):
            fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
            for e in range(10):
                ax = axes[e // 2, e % 2]
                ax.plot(time_ms, ecg_test[h, :, e] * 1000, 'k-', lw=1, label='True')
                ax.plot(time_ms, pred_phy[h, :, e] * 1000, 'r--', lw=1, label='Pred')
                ax.set_ylabel(f'{electrode_names[e]}\n(\u00b5V)', fontsize=8)
                ax.grid(True, alpha=0.3)
                if e == 0:
                    ax.legend(fontsize=7)
            axes[-1, 0].set_xlabel('Time (ms)')
            axes[-1, 1].set_xlabel('Time (ms)')
            fig.suptitle(f'Raw Electrodes \u2014 {case_names[num_train + num_val + h]} ({model_name})',
                         fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'electrodes_heart{h}.png'), dpi=150)
            plt.close()

        # --- Plot: 12-lead clinical layout (3x4 + rhythm strip) ---
        layout = [
            ["I",   "aVR", "V1", "V4"],
            ["II",  "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        for h in range(num_viz):
            pred_12 = to_12lead(pred_phy[h])
            true_12 = to_12lead(ecg_test[h])
            true_dict = {n: true_12[:, j] for j, n in enumerate(lead_names)}
            pred_dict = {n: pred_12[:, j] for j, n in enumerate(lead_names)}

            fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
            for row_idx, row in enumerate(layout):
                for col_idx, ln in enumerate(row):
                    ax = axes[row_idx, col_idx]
                    ax.plot(time_ms, true_dict[ln] * 1000, 'k-', lw=0.8, label='True')
                    ax.plot(time_ms, pred_dict[ln] * 1000, 'r--', lw=0.8, label='Pred')
                    ax.set_title(ln, fontsize=10, pad=2)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel('\u00b5V', fontsize=7)
                    if row_idx == 0 and col_idx == 0:
                        ax.legend(fontsize=7)

            for ax in axes[3, :]:
                ax.set_visible(False)
            ax_rhythm = fig.add_subplot(4, 1, 4)
            ax_rhythm.plot(time_ms, true_dict["II"] * 1000, 'k-', lw=0.8, label='True')
            ax_rhythm.plot(time_ms, pred_dict["II"] * 1000, 'r--', lw=0.8, label='Pred')
            ax_rhythm.set_title('II (Rhythm Strip)', fontsize=10, pad=2)
            ax_rhythm.set_xlabel('Time (ms)')
            ax_rhythm.set_ylabel('\u00b5V', fontsize=7)
            ax_rhythm.grid(True, alpha=0.3)
            ax_rhythm.legend(fontsize=7)

            case_label = case_names[num_train + num_val + h]
            fig.suptitle(f'12-Lead ECG \u2014 {case_label} ({model_name})', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'ecg_12lead_heart{h}.png'), dpi=150)
            plt.close()

        print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    main()
