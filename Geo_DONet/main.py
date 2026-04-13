"""
Geo-DyNet: geometry → V_m(x,t) on the canonical reference.

DeepONet with 1 branch + 1 trunk:
  - Branch (geometry): theta (N, 60) PCA coefficients → (N, p)
  - Trunk (spatiotemporal): (ab, rt, tm, tv, t) on fixed M×T grid → (M×T, p)
    Time input is Fourier-encoded: t → [sin(ω₁t), cos(ω₁t), ..., sin(ωₖt), cos(ωₖt)]
    to help the MLP capture the sharp AP upstroke.
    Trunk input dim = 4 (Cobiveco) + 2×n_freq (Fourier time features).
  - Output: einsum(branch, trunk) → V_m(x,t) on reference mesh (N, M, T)

Temporal chunking: the full M×T grid (~6M points at 121 frames) is split
into chunks of T_c frames. The trunk is evaluated per chunk, and the
branch output is reused across chunks.

Data: geo_dynet_data.npz from package_training_data.py

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DyNet

    # Train
    python main.py --epochs 5000 --device cuda

    # Full temporal resolution
    python main.py --epochs 50000 --device cuda

    # Evaluate
    python main.py --test-model 1 --device cuda
"""
import os
import torch
import time as timer
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from opnn import *
import matplotlib.pyplot as plt
import random

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precompute_trunk_chunks(coords, time_points, chunk_size, device):
    """
    Precompute the (M*T_c, 5) trunk input grids for all temporal chunks.
    Called once at startup — avoids rebuilding every forward pass.
    Returns: list of (M*T_c, 5) tensors on device
    """
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
    Forward pass with precomputed temporal chunks.
    f_geo:        (N, geo_modes) — geometry branch input
    trunk_chunks: list of (M*T_c, 5) precomputed trunk grids on device
    M:            number of spatial nodes
    Returns: (N, M, T)
    """
    N = f_geo.shape[0]

    # Geometry branch — compute once
    y_br = model._branch_g(f_geo)  # (N, p)

    outputs = []
    for xt_chunk in trunk_chunks:
        T_c = xt_chunk.shape[0] // M
        y_tr = model.encode_trunk(xt_chunk)  # (M*T_c, p)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr)  # (N, M*T_c)
        outputs.append(y_out.reshape(N, M, T_c))

    return torch.cat(outputs, dim=2)  # (N, M, T)


def main():
    set_seed(42)

    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    width = args.width
    batch_size = args.batch_size

    # --- Configuration ---
    num_geomode = 60
    dim_br_geo = [num_geomode] + [width] * 4  # branch: theta → (N, width)
    dim_tr = [5] + [width] * 4                # trunk: (cobiveco, t) → (Q, width)

    learning_rate = 0.001 if args.lr_schedule else 0.0005
    frame_step = 1  # data already subsampled at packaging time
    time_chunk_size = 25  # frames per chunk during forward pass

    lr_tag = '_lrsched' if args.lr_schedule else ''
    save_directory = f'geo_donet_{epochs}ep_w{width}{lr_tag}'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = args.model_path or f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## Load Dataset
    data_path = os.path.join(DATA_BASE, "geo_dynet_data.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        print("Run: python package_training_data.py --frame-step 5")
        return

    dataset = np.load(data_path, allow_pickle=True)
    theta = dataset['theta'][:, :num_geomode].astype(np.float32)  # (125, 60)
    coords = dataset['coords'].astype(np.float32)                  # (50797, 4)
    vm_all = dataset['vm'].astype(np.float32)                      # (125, 50797, T_full)
    time_ms = dataset['time'].astype(np.float32)                   # (T_full,)
    case_names = dataset['case_names']

    # Subsample time if needed
    vm_all = vm_all[:, :, ::frame_step]
    time_ms = time_ms[::frame_step]

    n_total, n_pts, n_time = vm_all.shape
    print(f"Data: {n_total} cases, {n_pts} nodes, {n_time} time steps")
    print(f"Time range: [{time_ms[0]:.0f}, {time_ms[-1]:.0f}] ms, step={time_ms[1]-time_ms[0]:.0f} ms")

    # Split
    num_train = 95
    num_val = 5
    num_test = n_total - num_train - num_val
    print(f"Split: {num_train} train / {num_val} val / {num_test} test")

    f_train = theta[:num_train]
    vm_train_raw = vm_all[:num_train]
    f_val = theta[num_train:num_train + num_val]
    vm_val_raw = vm_all[num_train:num_train + num_val]
    f_test = theta[num_train + num_val:]
    vm_test_raw = vm_all[num_train + num_val:]

    ## Normalization
    # Trunk spatial: per-feature min-max on Cobiveco
    coords_t = torch.tensor(coords, dtype=torch.float).to(device)
    x_min = coords_t.min(dim=0, keepdim=True)[0]
    x_max = coords_t.max(dim=0, keepdim=True)[0]
    coords_norm = (coords_t - x_min) / (x_max - x_min + 1e-8)

    # Time: normalize to [0, 1]
    t_min, t_max = time_ms.min(), time_ms.max()
    time_norm = (time_ms - t_min) / (t_max - t_min + 1e-8)
    time_t = torch.tensor(time_norm, dtype=torch.float).to(device)

    # Geometry: z-score
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_norm = (f_train - f_mean) / f_std
    f_val_norm = (f_val - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

    # V_m: min-shift + std-scale (training stats only)
    vm_mean = vm_train_raw.min()
    vm_std = vm_train_raw.std()
    vm_train_norm = (vm_train_raw - vm_mean) / vm_std
    vm_val_norm = (vm_val_raw - vm_mean) / vm_std
    vm_test_norm = (vm_test_raw - vm_mean) / vm_std

    ## Tensors
    f_train_tensor = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    f_val_tensor = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)

    # V_m targets: 95 × 50797 × 121 × 4 bytes ≈ 2.3 GB — fits on A100 40GB
    vm_train_tensor = torch.tensor(vm_train_norm, dtype=torch.float).to(device)
    vm_val_tensor = torch.tensor(vm_val_norm, dtype=torch.float).to(device)

    # Precompute trunk chunks once (avoids rebuilding every forward pass)
    print("Precomputing trunk chunks...", flush=True)
    trunk_chunks = precompute_trunk_chunks(coords_norm, time_t,
                                           time_chunk_size, device)
    print(f"  {len(trunk_chunks)} chunks, {sum(c.shape[0] for c in trunk_chunks)} total trunk points")

    model = opnn(dim_br_geo, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.lr_schedule:
        from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
        sched1 = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=2500)
        sched2 = ConstantLR(optimizer, factor=1.0, total_iters=epochs)
        scheduler = SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[2500])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: geo {dim_br_geo}, trunk {dim_tr}, params {n_params:,}")
    print(f"Temporal chunks: {time_chunk_size} frames/chunk")

    if test_model == 0:
        print(f"--- Starting Training ---", flush=True)

        train_loss_his, test_loss_his = [], []
        best_val_loss = float('inf')

        # Simple index-based batching (no DataLoader to control memory)
        train_indices = np.arange(num_train)

        tic = timer.time()
        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_indices)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, num_train, batch_size):
                b_idx = train_indices[b_start:b_start + batch_size]
                b_f = f_train_tensor[b_idx]
                b_vm = vm_train_tensor[b_idx]  # already on GPU

                optimizer.zero_grad()
                pred = chunked_forward(model, b_f, trunk_chunks, n_pts)
                loss = ((pred - b_vm) ** 2).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_loss_his.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                pred_val = chunked_forward(model, f_val_tensor, trunk_chunks, n_pts)
                val_loss = ((pred_val - vm_val_tensor) ** 2).mean().item()
                test_loss_his.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if args.lr_schedule:
                scheduler.step()

            if epoch % 10 == 0:
                elapsed = timer.time() - tic
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                print(f'Epoch: {epoch}/{epochs} | Train: {avg_train_loss:.6f} | '
                      f'Val: {val_loss:.6f} | Best Val: {best_val_loss:.6f} | '
                      f'ETA: {int(eta//60)}m{int(eta%60)}s',
                      flush=True)

        total_min = int((timer.time() - tic) / 60)
        print(f"Total training time: {total_min} min")

        np.savetxt(f'./Predictions/{save_directory}/train_loss.txt',
                   np.array(train_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/test_loss.txt',
                   np.array(test_loss_his))

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss_his, label='Train', alpha=0.8)
        ax.semilogy(test_loss_his, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{save_directory}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./Predictions/{save_directory}/loss_curve.png', dpi=150)
        plt.close()
        print(f"Saved loss curve to Predictions/{save_directory}/loss_curve.png")

    else:
        # --- Evaluation ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Generating Evaluation ---")

        with torch.no_grad():
            # Predict test set in single-heart chunks to save memory
            # Warm up GPU (first call has overhead)
            _ = chunked_forward(model, f_test_tensor[:1], trunk_chunks, n_pts)
            torch.cuda.synchronize()

            all_pred = []
            infer_times = []
            for i in range(num_test):
                f_i = f_test_tensor[i:i + 1]
                torch.cuda.synchronize()
                t0 = timer.time()
                pred_i = chunked_forward(model, f_i, trunk_chunks, n_pts)
                torch.cuda.synchronize()
                t1 = timer.time()
                infer_times.append(t1 - t0)
                all_pred.append(to_numpy(pred_i[0]))  # (M, T)
            pred_test = np.stack(all_pred)  # (num_test, M, T)

            print(f"Inference time per case: {np.mean(infer_times)*1000:.1f} +/- "
                  f"{np.std(infer_times)*1000:.1f} ms "
                  f"(total {np.sum(infer_times):.2f} s for {num_test} cases)")

            # Denormalize
            pred_phy = pred_test * vm_std + vm_mean

            # Per-case error metrics
            print(f"\n{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>10}")
            print("-" * 58)
            l2_errors, mae_errors = [], []
            for i in range(num_test):
                u_p = pred_phy[i]       # (M, T)
                u_t = vm_test_raw[i]    # (M, T)
                l2 = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                mae = np.mean(np.abs(u_p - u_t))
                l2_errors.append(l2)
                mae_errors.append(mae)
                print(f"{case_names[num_train + num_val + i]:<35}"
                      f" {l2:10.4f} {mae:10.2f}")

            print("-" * 58)
            print(f"{'Mean':<35} {np.mean(l2_errors):10.4f} "
                  f"{np.mean(mae_errors):10.2f}")
            print(f"{'Std':<35} {np.std(l2_errors):10.4f} "
                  f"{np.std(mae_errors):10.2f}")

            # Plot V_m traces at a few spatial points for first 2 test hearts
            dimon_data = np.load(os.path.join(
                os.path.dirname(__file__), "..", "DIMON_training_data_healthy.npz"))
            cartesian_coords = dimon_data['cartesian_coords']

            num_viz = min(2, num_test)
            sample_nodes = np.linspace(0, n_pts - 1, 5, dtype=int)  # 5 nodes
            time_axis = time_ms  # physical ms

            for h in range(num_viz):
                fig, axes = plt.subplots(len(sample_nodes), 1,
                                         figsize=(10, 2.5 * len(sample_nodes)),
                                         sharex=True)
                for j, node in enumerate(sample_nodes):
                    ax = axes[j]
                    ax.plot(time_axis, vm_test_raw[h, node, :],
                            'k-', linewidth=1, label='True')
                    ax.plot(time_axis, pred_phy[h, node, :],
                            'r--', linewidth=1, label='Pred')
                    ax.set_ylabel(f'Node {node}\n(mV)', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if j == 0:
                        ax.legend(fontsize=8)
                axes[-1].set_xlabel('Time (ms)')
                fig.suptitle(f'V_m traces — {case_names[num_train + num_val + h]}',
                             fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(dump_test, f'vm_traces_heart{h}.png'),
                            dpi=150)
                plt.close()

            # 3D whole-heart V_m snapshots for first 2 test hearts, first 20 frames
            num_viz_hearts = min(2, num_test)
            num_viz_frames = 20
            vm_min, vm_max = -90, 50  # fixed color scale (mV)

            snap_dir = os.path.join(dump_test, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            print(f"Saving whole-heart snapshots to {snap_dir}")

            for h in range(num_viz_hearts):
                case_label = case_names[num_train + num_val + h]
                for fi in range(min(num_viz_frames, pred_phy.shape[2])):
                    t_val = time_ms[fi]
                    u_p = pred_phy[h, :, fi]
                    u_t = vm_test_raw[h, :, fi]

                    fig = plt.figure(figsize=(18, 6))

                    ax1 = fig.add_subplot(131, projection='3d')
                    sc1 = ax1.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=u_t, cmap='jet', s=1,
                                      vmin=vm_min, vmax=vm_max)
                    ax1.set_title(f"True (t={t_val:.0f} ms)")
                    plt.colorbar(sc1, ax=ax1, shrink=0.5)

                    ax2 = fig.add_subplot(132, projection='3d')
                    sc2 = ax2.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=u_p, cmap='jet', s=1,
                                      vmin=vm_min, vmax=vm_max)
                    ax2.set_title(f"Pred (t={t_val:.0f} ms)")
                    plt.colorbar(sc2, ax=ax2, shrink=0.5)

                    ax3 = fig.add_subplot(133, projection='3d')
                    err = np.abs(u_t - u_p)
                    sc3 = ax3.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=err, cmap='Reds', s=1,
                                      vmin=0, vmax=20)
                    ax3.set_title(f"Abs Error (Mean: {err.mean():.2f} mV)")
                    plt.colorbar(sc3, ax=ax3, shrink=0.5)

                    for ax in [ax1, ax2, ax3]:
                        ax.set_axis_off()
                    plt.suptitle(f"{case_label} — t={t_val:.0f} ms", fontsize=11)
                    plt.tight_layout()
                    plt.savefig(os.path.join(snap_dir,
                                f"heart{h}_frame{fi:03d}_t{t_val:.0f}ms.png"),
                                dpi=150)
                    plt.close()

            # Save predictions
            np.savez_compressed(
                os.path.join(dump_test, "test_predictions.npz"),
                pred=pred_phy, true=vm_test_raw,
                case_names=case_names[num_train + num_val:])
            print(f"\nEvaluation complete. Plots in {dump_test}")


if __name__ == "__main__":
    main()
