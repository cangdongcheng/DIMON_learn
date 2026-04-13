"""
Geo-DeepONet: geometry → activation time map on the canonical reference.
Adapted from Cobiveco_with_fiber/main.py with pacing branch removed.

Single fixed 7-node Durrer activation — geometry is the only input variable.

Input:  theta (N, 60) PCA geometry + Cobiveco (M, 4) trunk
Output: AT(x) on reference mesh (N, M)

Data:   geo_deeponet_data.npz from package_training_data.py

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DeepONet

    # Train
    python main.py --epochs 50000 --device cuda

    # Evaluate saved model
    python main.py --test-model 1 --device cuda
"""
import os
import torch
import time
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


def main():
    set_seed(42)

    ## 1. Hyperparameters & Configuration
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model

    normalize = 1

    # Architecture dimensions
    num_geomode = 60
    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_tr = [4, 200, 200, 200, 200]  # 4D Cobiveco trunk

    batch_size = 10
    learning_rate = 0.0005

    save_directory = f'cobiveco_{normalize}norm_{epochs}ep_{learning_rate}lr'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## 2. Load Dataset
    dataset = np.load(os.path.join(DATA_BASE, "geo_deeponet_data.npz"),
                      allow_pickle=True)
    theta = dataset['theta'][:, :num_geomode]  # (125, 60)
    x_data = dataset['coords']                  # (50797, 4) Cobiveco
    u_all = dataset['at']                       # (125, 50797)
    case_names = dataset['case_names']

    num_train_hearts = 95
    num_val_hearts = 5
    num_test_hearts = 25

    num_pts = x_data.shape[0]
    print(f"Data: {theta.shape[0]} cases, {num_pts} nodes, {num_geomode} PCA modes")
    print(f"Split: {num_train_hearts} train / {num_val_hearts} val / {num_test_hearts} test")

    ## Split
    f_train = theta[:num_train_hearts]
    u_train_raw = u_all[:num_train_hearts]

    f_val = theta[num_train_hearts:num_train_hearts + num_val_hearts]
    u_val_raw = u_all[num_train_hearts:num_train_hearts + num_val_hearts]

    f_test = theta[num_train_hearts + num_val_hearts:
                   num_train_hearts + num_val_hearts + num_test_hearts]
    u_test_raw = u_all[num_train_hearts + num_val_hearts:
                       num_train_hearts + num_val_hearts + num_test_hearts]

    x_tensor_raw = torch.tensor(x_data, dtype=torch.float).to(device)

    ## 3. Normalization Logic
    if normalize == 1:
        # Normalize Trunk Input (per-feature min-max)
        x_min = x_tensor_raw.min(dim=0, keepdim=True)[0]
        x_max = x_tensor_raw.max(dim=0, keepdim=True)[0]
        x_tensor = (x_tensor_raw - x_min) / (x_max - x_min + 1e-8)

        # Normalize Branch Geometry (z-score)
        f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
        f_train_norm = (f_train - f_mean) / f_std
        f_val_norm = (f_val - f_mean) / f_std
        f_test_norm = (f_test - f_mean) / f_std

        # Normalize Labels (min-shift + std-scale, training stats only)
        u_mean_train = u_train_raw.min()
        u_std_train = u_train_raw.std()
        u_train_norm = (u_train_raw - u_mean_train) / u_std_train
        u_val_norm = (u_val_raw - u_mean_train) / u_std_train
        u_test_norm = (u_test_raw - u_mean_train) / u_std_train
    else:
        x_tensor = x_tensor_raw
        f_train_norm, f_val_norm, f_test_norm = f_train, f_val, f_test
        u_train_norm, u_val_norm, u_test_norm = u_train_raw, u_val_raw, u_test_raw
        u_mean_train, u_std_train = 0.0, 1.0

    ## 4. Tensors & DataLoader
    f_train_tensor = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    u_train_tensor = torch.tensor(u_train_norm, dtype=torch.float).to(device)
    f_val_tensor = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    u_val_tensor = torch.tensor(u_val_norm, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)
    u_test_tensor = torch.tensor(u_test_norm, dtype=torch.float).to(device)

    model = opnn(dim_br_geo, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: geo {dim_br_geo}, trunk {dim_tr}, params {n_params:,}")

    if test_model == 0:
        print(f"--- Starting Training (Batch Size: {batch_size} hearts) ---")
        print(f"Trunk Input Dimension: {x_data.shape[1]}")

        train_ds = TensorDataset(f_train_tensor, u_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        train_loss_his, test_loss_his = [], []
        best_val_loss = float('inf')

        tic = time.time()
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for b_f, b_u in train_loader:
                optimizer.zero_grad()
                y_pred = model.forward(b_f, x_tensor)
                loss = ((y_pred - b_u) ** 2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_loss_his.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                y_val = model.forward(f_val_tensor, x_tensor)
                val_loss = ((y_val - u_val_tensor) ** 2).mean().item()
                y_test = model.forward(f_test_tensor, x_tensor)
                test_loss = ((y_test - u_test_tensor) ** 2).mean().item()
                test_loss_his.append(test_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if epoch % 100 == 0:
                # Compute physical MAE for monitoring
                y_test_phy = to_numpy(y_test) * u_std_train + u_mean_train
                mae = np.abs(y_test_phy - u_test_raw).mean()
                print(f'Epoch: {epoch} | Train: {avg_train_loss:.6f} | '
                      f'Val: {val_loss:.6f} | Best Val: {best_val_loss:.6f} | '
                      f'Test: {test_loss:.6f} | MAE: {mae:.2f} ms', flush=True)

        print(f"Total training time: {int((time.time() - tic) / 60)} min")
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
        # --- 5. Evaluation ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Generating Evaluation ---")
        num_viz_hearts = 2

        # Load cartesian coords for 3D plotting
        dimon_data = np.load(os.path.join(os.path.dirname(__file__), "..", "DIMON_training_data_healthy.npz"))
        cartesian_coords = dimon_data['cartesian_coords']  # (50797, 3)

        with torch.no_grad():
            # Denormalize predictions
            y_pred_test_raw = model.forward(f_test_tensor, x_tensor)
            u_test_pred = to_numpy(y_pred_test_raw) * u_std_train + u_mean_train

            f_train_subset = f_train_tensor[:num_viz_hearts]
            y_pred_train_raw = model.forward(f_train_subset, x_tensor)
            u_train_pred = to_numpy(y_pred_train_raw) * u_std_train + u_mean_train

            u_train_phy = u_train_raw[:num_viz_hearts]
            u_test_phy_viz = u_test_raw[:num_viz_hearts]

            # 3D scatter comparison plots
            def plot_comparison(pred_batch, true_batch, folder_name, file_prefix):
                os.makedirs(folder_name, exist_ok=True)
                for i in range(min(num_viz_hearts, pred_batch.shape[0])):
                    u_p = pred_batch[i]
                    u_t = true_batch[i]
                    v_min = min(u_t.min(), u_p.min())
                    v_max = max(u_t.max(), u_p.max())

                    fig = plt.figure(figsize=(18, 6))

                    ax1 = fig.add_subplot(131, projection='3d')
                    sc1 = ax1.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=u_t, cmap='jet', s=1,
                                      vmin=v_min, vmax=v_max)
                    ax1.set_title(f"True (Heart {i})")
                    plt.colorbar(sc1, ax=ax1, shrink=0.5)

                    ax2 = fig.add_subplot(132, projection='3d')
                    sc2 = ax2.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=u_p, cmap='jet', s=1,
                                      vmin=v_min, vmax=v_max)
                    ax2.set_title(f"Pred (Heart {i})")
                    plt.colorbar(sc2, ax=ax2, shrink=0.5)

                    ax3 = fig.add_subplot(133, projection='3d')
                    err = np.abs(u_t - u_p)
                    sc3 = ax3.scatter(cartesian_coords[:, 0],
                                      cartesian_coords[:, 1],
                                      cartesian_coords[:, 2],
                                      c=err, cmap='Reds', s=1)
                    ax3.set_title(f"Abs Error (Mean: {np.mean(err):.2f} ms)")
                    plt.colorbar(sc3, ax=ax3, shrink=0.5)

                    for ax in [ax1, ax2, ax3]:
                        ax.set_axis_off()
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_name,
                                f"{file_prefix}_heart{i}.png"), dpi=200)
                    plt.close()

            print(f"Saving plots to {dump_train} and {dump_test}")
            plot_comparison(u_train_pred, u_train_phy, dump_train, 'train')
            plot_comparison(u_test_pred, u_test_phy_viz, dump_test, 'test')

            # --- 6. Error Statistics ---
            print("\nCalculating error statistics...")
            num_test = u_test_pred.shape[0]

            print(f"\n{'Case':<35} {'Rel L2':>10} {'MAE (ms)':>10}")
            print("-" * 58)
            l2_errors, mae_errors = [], []
            for i in range(num_test):
                u_p = u_test_pred[i]
                u_t = u_test_raw[i]
                l2_err = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                mae_err = np.mean(np.abs(u_p - u_t))
                l2_errors.append(l2_err)
                mae_errors.append(mae_err)
                print(f"{case_names[num_train_hearts + num_val_hearts + i]:<35}"
                      f" {l2_err:10.4f} {mae_err:10.2f}")

            print("-" * 58)
            print(f"{'Mean':<35} {np.mean(l2_errors):10.4f} "
                  f"{np.mean(mae_errors):10.2f}")
            print(f"{'Std':<35} {np.std(l2_errors):10.4f} "
                  f"{np.std(mae_errors):10.2f}")

            # Save predictions
            np.savez_compressed(
                os.path.join(dump_test, "test_predictions.npz"),
                pred=u_test_pred, true=u_test_raw,
                case_names=case_names[num_train_hearts + num_val_hearts:])
            print(f"\nEvaluation complete.")


if __name__ == "__main__":
    main()
