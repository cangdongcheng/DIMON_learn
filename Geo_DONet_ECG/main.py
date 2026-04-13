"""
Geo-DONet-ECG (Approach C): end-to-end geometry → V_m(x,t) + ECG(10,t).

Joint training with two supervised outputs:
  loss = loss_vm + lambda_ecg * loss_ecg

Branch (geometry): theta (60D PCA) → (N, p)
Trunk (spatiotemporal): (cobiveco, t) + Fourier encoding → (Q, p)
V_m: einsum(branch, trunk) → (N, M, T)
ECG: V_m → learned transfer → (N, T, 10)

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DONet_ECG

    python main.py --epochs 2000 --device cuda
    python main.py --test-model 1 --epochs 2000 --device cuda
"""
import os
import torch
import time as timer
import numpy as np
import matplotlib.pyplot as plt
import random

from utils import *
from opnn import GeoDONetECG

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precompute_trunk_chunks(coords, time_points, chunk_size, device):
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
    """Returns (N, M, T) V_m prediction."""
    N = f_geo.shape[0]
    y_br = model._branch_g(f_geo)

    outputs = []
    for xt_chunk in trunk_chunks:
        T_c = xt_chunk.shape[0] // M
        y_tr = model.encode_trunk(xt_chunk)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr)
        outputs.append(y_out.reshape(N, M, T_c))

    return torch.cat(outputs, dim=2)  # (N, M, T)


def main():
    set_seed(42)

    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model

    # --- Configuration ---
    num_geomode = 60
    dim_br_geo = [num_geomode, 300, 300, 300, 300]
    dim_tr = [5, 300, 300, 300, 300]  # trunk: (cobiveco, t) → (Q, 300)

    batch_size = 24
    learning_rate = 0.0005
    lambda_ecg = 1.0  # weight for ECG loss relative to V_m loss
    time_chunk_size = 25

    save_directory = f'geo_donet_ecg_{epochs}ep_{learning_rate}lr_lambda{lambda_ecg}'

    dump_test = f'./Predictions/{save_directory}/Test/'
    model_path = f'CheckPts/model_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## Load data
    data = np.load(os.path.join(DATA_BASE, "geo_dynet_data.npz"), allow_pickle=True)
    theta = data['theta'][:, :num_geomode].astype(np.float32)
    coords = data['coords'].astype(np.float32)
    vm_all = data['vm'].astype(np.float32)       # (125, 50797, 121)
    ecg_all = data['ecg'].astype(np.float32)      # (125, 121, 10)
    time_ms = data['time'].astype(np.float32)
    case_names = data['case_names']

    n_total, n_pts, n_time = vm_all.shape
    n_electrodes = ecg_all.shape[2]
    print(f"Data: {n_total} cases, {n_pts} nodes, {n_time} frames, {n_electrodes} electrodes")

    ## Split
    num_train, num_val = 95, 5
    num_test = n_total - num_train - num_val

    f_train = theta[:num_train]
    vm_train_raw = vm_all[:num_train]
    ecg_train_raw = ecg_all[:num_train]

    f_val = theta[num_train:num_train + num_val]
    vm_val_raw = vm_all[num_train:num_train + num_val]
    ecg_val_raw = ecg_all[num_train:num_train + num_val]

    f_test = theta[num_train + num_val:]
    vm_test_raw = vm_all[num_train + num_val:]
    ecg_test_raw = ecg_all[num_train + num_val:]

    print(f"Split: {num_train} train / {num_val} val / {num_test} test")

    ## Normalization
    # Trunk spatial: per-feature min-max
    coords_t = torch.tensor(coords, dtype=torch.float).to(device)
    x_min = coords_t.min(dim=0, keepdim=True)[0]
    x_max = coords_t.max(dim=0, keepdim=True)[0]
    coords_norm = (coords_t - x_min) / (x_max - x_min + 1e-8)

    # Time: [0, 1]
    t_min, t_max = time_ms.min(), time_ms.max()
    time_norm = (time_ms - t_min) / (t_max - t_min + 1e-8)
    time_t = torch.tensor(time_norm, dtype=torch.float).to(device)

    # Geometry: z-score
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_n = (f_train - f_mean) / f_std
    f_val_n = (f_val - f_mean) / f_std
    f_test_n = (f_test - f_mean) / f_std

    # V_m: min + std
    vm_mean = vm_train_raw.min()
    vm_std = vm_train_raw.std()
    vm_train_n = (vm_train_raw - vm_mean) / vm_std
    vm_val_n = (vm_val_raw - vm_mean) / vm_std

    # ECG: min + std
    ecg_mean = ecg_train_raw.min()
    ecg_std_val = ecg_train_raw.std()
    ecg_train_n = (ecg_train_raw - ecg_mean) / ecg_std_val
    ecg_val_n = (ecg_val_raw - ecg_mean) / ecg_std_val

    ## Tensors on GPU
    f_train_t = torch.tensor(f_train_n, dtype=torch.float).to(device)
    f_val_t = torch.tensor(f_val_n, dtype=torch.float).to(device)
    f_test_t = torch.tensor(f_test_n, dtype=torch.float).to(device)
    vm_train_t = torch.tensor(vm_train_n, dtype=torch.float).to(device)
    vm_val_t = torch.tensor(vm_val_n, dtype=torch.float).to(device)
    ecg_train_t = torch.tensor(ecg_train_n, dtype=torch.float).to(device)
    ecg_val_t = torch.tensor(ecg_val_n, dtype=torch.float).to(device)

    # Precompute trunk chunks
    print("Precomputing trunk chunks...", flush=True)
    trunk_chunks = precompute_trunk_chunks(coords_norm, time_t, time_chunk_size, device)
    print(f"  {len(trunk_chunks)} chunks")

    ## Model
    model = GeoDONetECG(dim_br_geo, dim_tr,
                        n_nodes=n_pts, n_electrodes=n_electrodes,
                        ecg_mode='linear').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Flat LR — cosine annealing caused instability at 0.001

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (incl ECG head), lambda_ecg={lambda_ecg}")
    print(f"LR: {learning_rate} (flat)")

    if test_model == 0:
        ## Training
        train_indices = np.arange(num_train)
        train_loss_his, val_loss_his = [], []
        best_val_loss = float('inf')

        print("--- Starting Training ---", flush=True)
        tic = timer.time()

        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_indices)
            epoch_loss_vm, epoch_loss_ecg = 0.0, 0.0
            n_batches = 0

            for b_start in range(0, num_train, batch_size):
                b_idx = train_indices[b_start:b_start + batch_size]
                b_f = f_train_t[b_idx]
                b_vm = vm_train_t[b_idx]        # (B, M, T)
                b_ecg = ecg_train_t[b_idx]      # (B, T, 10)

                optimizer.zero_grad()

                # V_m prediction
                vm_pred = chunked_forward(model, b_f, trunk_chunks, n_pts)
                loss_vm = ((vm_pred - b_vm) ** 2).mean()

                # ECG prediction from V_m
                ecg_pred = model.predict_ecg(vm_pred)  # (B, T, 10)
                loss_ecg = ((ecg_pred - b_ecg) ** 2).mean()

                loss = loss_vm + lambda_ecg * loss_ecg
                loss.backward()
                optimizer.step()

                epoch_loss_vm += loss_vm.item()
                epoch_loss_ecg += loss_ecg.item()
                n_batches += 1

            avg_vm = epoch_loss_vm / n_batches
            avg_ecg = epoch_loss_ecg / n_batches
            train_loss_his.append(avg_vm + lambda_ecg * avg_ecg)

            # Validation
            model.eval()
            with torch.no_grad():
                vm_val_pred = chunked_forward(model, f_val_t, trunk_chunks, n_pts)
                val_vm = ((vm_val_pred - vm_val_t) ** 2).mean().item()
                ecg_val_pred = model.predict_ecg(vm_val_pred)
                val_ecg = ((ecg_val_pred - ecg_val_t) ** 2).mean().item()
                val_total = val_vm + lambda_ecg * val_ecg
                val_loss_his.append(val_total)

            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if epoch % 10 == 0:
                elapsed = timer.time() - tic
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                print(f'Epoch {epoch}/{epochs} | VM: {avg_vm:.6f} | '
                      f'ECG: {avg_ecg:.6f} | Val: {val_total:.6f} | '
                      f'Best: {best_val_loss:.6f} | '
                      f'ETA: {int(eta//60)}m{int(eta%60)}s',
                      flush=True)

        print(f"Training done: {int((timer.time() - tic) / 60)} min")
        np.savetxt(os.path.join(f'Predictions/{save_directory}', 'train_loss.txt'),
                   train_loss_his)
        np.savetxt(os.path.join(f'Predictions/{save_directory}', 'val_loss.txt'),
                   val_loss_his)

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss_his, label='Train', alpha=0.8)
        ax.semilogy(val_loss_his, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (VM + ECG)')
        ax.set_title(f'{save_directory}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(f'Predictions/{save_directory}', 'loss_curve.png'), dpi=150)
        plt.close()
        print(f"Saved loss curve to Predictions/{save_directory}/loss_curve.png")

    else:
        ## Evaluation
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Evaluation ---")
        electrode_names = ["LA", "RA", "LL", "RL", "V1", "V2", "V3", "V4", "V5", "V6"]

        with torch.no_grad():
            # Inference with timing
            _ = chunked_forward(model, f_test_t[:1], trunk_chunks, n_pts)
            torch.cuda.synchronize()

            all_vm, all_ecg = [], []
            infer_times = []
            for i in range(num_test):
                f_i = f_test_t[i:i + 1]
                torch.cuda.synchronize()
                t0 = timer.time()
                vm_i = chunked_forward(model, f_i, trunk_chunks, n_pts)
                ecg_i = model.predict_ecg(vm_i)
                torch.cuda.synchronize()
                t1 = timer.time()
                infer_times.append(t1 - t0)
                all_vm.append(to_numpy(vm_i[0]))
                all_ecg.append(to_numpy(ecg_i[0]))

            pred_vm = np.stack(all_vm) * vm_std + vm_mean
            pred_ecg = np.stack(all_ecg) * ecg_std_val + ecg_mean

            print(f"Inference: {np.mean(infer_times)*1000:.1f} +/- "
                  f"{np.std(infer_times)*1000:.1f} ms/case")

            # V_m metrics
            print(f"\n--- V_m Errors ---")
            print(f"{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>10}")
            print("-" * 58)
            for i in range(num_test):
                p, t = pred_vm[i], vm_test_raw[i]
                l2 = np.linalg.norm(p - t) / np.linalg.norm(t)
                mae = np.mean(np.abs(p - t))
                print(f"{case_names[num_train + num_val + i]:<35} {l2:10.4f} {mae:10.2f}")

            # ECG metrics
            print(f"\n--- ECG Errors ---")
            print(f"{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>12}")
            print("-" * 60)
            for i in range(num_test):
                p, t = pred_ecg[i], ecg_test_raw[i]
                l2 = np.linalg.norm(p - t) / np.linalg.norm(t)
                mae = np.mean(np.abs(p - t))
                print(f"{case_names[num_train + num_val + i]:<35} {l2:10.4f} {mae:12.6f}")

            # Plot ECG traces for first 2 test hearts
            num_viz = min(2, num_test)
            for h in range(num_viz):
                fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
                for e in range(10):
                    ax = axes[e // 2, e % 2]
                    ax.plot(time_ms, ecg_test_raw[h, :, e] * 1000, 'k-', lw=1, label='True')
                    ax.plot(time_ms, pred_ecg[h, :, e] * 1000, 'r--', lw=1, label='Pred')
                    ax.set_ylabel(f'{electrode_names[e]}\n(µV)', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if e == 0:
                        ax.legend(fontsize=7)
                axes[-1, 0].set_xlabel('Time (ms)')
                axes[-1, 1].set_xlabel('Time (ms)')
                fig.suptitle(f'ECG — {case_names[num_train + num_val + h]}', fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(dump_test, f'ecg_heart{h}.png'), dpi=150)
                plt.close()

            # Save
            np.savez_compressed(os.path.join(dump_test, "predictions.npz"),
                                pred_vm=pred_vm, true_vm=vm_test_raw,
                                pred_ecg=pred_ecg, true_ecg=ecg_test_raw,
                                case_names=case_names[num_train + num_val:])
            print(f"\nEvaluation complete. Plots in {dump_test}")


if __name__ == "__main__":
    main()
