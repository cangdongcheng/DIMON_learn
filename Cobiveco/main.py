"""
Dongcheng Cang, dccang@u.nus.edu
Replication of Failed Baseline: Flattened Data + Pure Data-Driven
"""

import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from utils import *
from opnn import *

def main():
    ## hyperparameters
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    save_step = args.save_step
    test_model = args.test_model
    ## load_path is set after save_directory is defined

    # --- Configuration ---
    num_geomode = 60  # Updated for new SSM modes
    if_normal = 0
    batch_size = 95
    u_scale = 1.0
    learning_rate = 0.0005

    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200]
    dim_tr = [4, 200, 200, 200, 200]

    # Split Definition (By Hearts)
    train_hearts = 95
    val_hearts = 5
    test_hearts = 25
    num_sims = 9

    save_directory = f'unorm_flat_cobiveco4d_{epochs}ep_{learning_rate}lr'

    ## create folders
    dump_test = f'./Predictions/{save_directory}/Test/'
    model_path = f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## 1. Load Dataset
    dataset = np.load("/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV/DIMON_training_data_healthy_fixed.npz")
    theta_all = dataset['theta']              # (125, 60)
    pacing_all = dataset['pacing']            # (9, 4)
    u_all = dataset['u_data']                 # (125, 9, 48287)
    x_data = dataset['cartesian_coords']      # (48287, 3)
    cobiveco = dataset['cobiveco']

    num_pts = x_data.shape[0]

    # --- Flattening Data ---
    theta_flat = np.repeat(theta_all, num_sims, axis=0)      # (1125, 60)
    pacing_flat = np.tile(pacing_all, (125, 1))              # (1125, 4)
    u_flat = u_all.reshape(-1, num_pts)                      # (1125, 48287)

    # --- Slicing Splits ---
    train_idx = train_hearts * num_sims
    val_idx = train_idx + (val_hearts * num_sims)

    f_train = theta_flat[:train_idx, :]
    x_pace_train = pacing_flat[:train_idx, :]
    u_train = u_flat[:train_idx, :]

    f_val = theta_flat[train_idx:val_idx, :]
    x_pace_val = pacing_flat[train_idx:val_idx, :]
    u_val = u_flat[train_idx:val_idx, :]

    f_test = theta_flat[val_idx:, :]
    x_pace_test = pacing_flat[val_idx:, :]
    u_test = u_flat[val_idx:, :]

    ## 2. Normalization
    u_test_phy = u_test.copy()
    u_val_phy = u_val.copy()
    if if_normal == 1:
        # Geo branch: z-score
        f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
        f_train = (f_train - f_mean) / f_std
        f_val = (f_val - f_mean) / f_std
        f_test = (f_test - f_mean) / f_std

        # Labels: min-shift + std scaling
        u_mean_train = u_train.min()
        u_std = u_train.std()
        u_train = (u_train - u_mean_train) / u_std
        u_val = (u_val - u_mean_train) / u_std
        u_test = (u_test - u_mean_train) / u_std
    else:
        u_mean_train = 0.0
        u_std = 1.0

    ## 3. Tensors & Loader
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_train_tensor = torch.tensor(f_train, dtype=torch.float).to(device)
    x_pace_train_tensor = torch.tensor(x_pace_train, dtype=torch.float).to(device)

    u_val_tensor = torch.tensor(u_val, dtype=torch.float).to(device)
    f_val_tensor = torch.tensor(f_val, dtype=torch.float).to(device)
    x_pace_val_tensor = torch.tensor(x_pace_val, dtype=torch.float).to(device)

    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    x_pace_test_tensor = torch.tensor(x_pace_test, dtype=torch.float).to(device)
    u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)

    # Trunk: min-max normalization
    x_tensor_raw = torch.tensor(cobiveco, dtype=torch.float).to(device)
    x_min = x_tensor_raw.min(dim=0, keepdim=True)[0]
    x_max = x_tensor_raw.max(dim=0, keepdim=True)[0]
    x_tensor = (x_tensor_raw - x_min) / (x_max - x_min + 1e-8)

    train_ds = TensorDataset(f_train_tensor, x_pace_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    if test_model == 0:
        train_loss = np.zeros((args.epochs, 1))
        val_loss_his = np.zeros((args.epochs, 1))
        best_val_loss = float('inf')

        ## 4. Training
        def train_epoch(epoch):
            model.train()
            epoch_loss = 0
            for b_f, b_pace, b_u in train_loader:
                optimizer.zero_grad()
                y_pred = model.forward(b_f, b_pace, x_tensor)
                loss = ((y_pred - b_u)**2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss[epoch, 0] = epoch_loss / len(train_loader)

        print(f'Start training (Train: {train_hearts*num_sims}, Val: {val_hearts*num_sims})...', flush=True)
        tic = time.time()
        for epoch in range(0, epochs):
            if epoch == 20000:
                for g in optimizer.param_groups: g['lr'] = 0.0001

            train_epoch(epoch)

            ## Validation
            model.eval()
            with torch.no_grad():
                y_val_raw = model.forward(f_val_tensor, x_pace_val_tensor, x_tensor)
                val_loss = ((y_val_raw - u_val_tensor)**2).mean().item()
                val_loss_his[epoch, 0] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_path)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Train MSE: {train_loss[epoch, 0]:.6f} | Val MSE: {val_loss:.6f}', flush=True)

        toc = time.time()
        print(f'Total training time: {int((toc-tic)/60)} min', flush=True)
        np.savetxt(f'./Predictions/{save_directory}/train_loss.txt', train_loss)
        np.savetxt(f'./Predictions/{save_directory}/val_loss.txt', val_loss_his)

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss.flatten(), label='Train', alpha=0.8)
        ax.semilogy(val_loss_his.flatten(), label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(save_directory)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./Predictions/{save_directory}/loss_curve.png', dpi=150)
        plt.close()
        print(f"Saved loss curve to Predictions/{save_directory}/loss_curve.png")

    else:
        # --- 5. Evaluation ---
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("Generating 3D point cloud visualizations...")
        num_viz_hearts = min(2, test_hearts) # Visualize first 2 hearts
            
        with torch.no_grad():
            y_pred_test_raw = model.forward(f_test_tensor, x_pace_test_tensor, x_tensor)
            u_test_pred_flat = to_numpy(y_pred_test_raw) * u_std + u_mean_train
                
            # Reshape back to (Hearts, Pacing Sites, Nodes)
            u_test_pred = u_test_pred_flat.reshape(test_hearts, num_sims, num_pts)
            u_test_phy_3d = u_test_phy.reshape(test_hearts, num_sims, num_pts)

            def plot_comparison(pred_batch, true_batch, folder_name, file_prefix):
                os.makedirs(folder_name, exist_ok=True)
                for i in range(num_viz_hearts):
                    for s in range(3): # Plot first 3 pacing sites per heart
                        u_p = pred_batch[i, s]
                        u_t = true_batch[i, s]
                        
                        v_min, v_max = min(u_t.min(), u_p.min()), max(u_t.max(), u_p.max())
                        fig = plt.figure(figsize=(18, 6))
                        
                        ax1 = fig.add_subplot(131, projection='3d')
                        sc1 = ax1.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=u_t, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax1.set_title(f"{file_prefix.capitalize()} True (Heart {i}, Sim {s})")
                        plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

                        ax2 = fig.add_subplot(132, projection='3d')
                        sc2 = ax2.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=u_p, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax2.set_title(f"{file_prefix.capitalize()} Pred (Heart {i}, Sim {s})")
                        plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

                        ax3 = fig.add_subplot(133, projection='3d')
                        err = np.abs(u_t - u_p)
                        sc3 = ax3.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=err, cmap='Reds', s=1, vmin=0)
                        ax3.set_title(f"Abs Error (Mean: {np.mean(err):.4f})")
                        plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

                        for ax in [ax1, ax2, ax3]: ax.set_axis_off()
                            
                        plt.tight_layout()
                        plt.savefig(os.path.join(folder_name, f"{file_prefix}_heart{i}_sim{s}_3d.png"), dpi=200)
                        plt.close()

            print(f"Saving test plots to {dump_test}")
            plot_comparison(u_test_pred, u_test_phy_3d, dump_test, 'test')
            
            # --- Calculate Pooled L2 and MAE Errors ---
            print("Calculating Pooled Evaluation Metrics...")
            l2_errors = []
            mae_errors = []

            for i in range(test_hearts):
                for s in range(num_sims):
                    u_p = u_test_pred[i, s]
                    u_t = u_test_phy_3d[i, s]
                    
                    # Relative L2 norm and MAE
                    l2_err = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                    mae_err = np.mean(np.abs(u_p - u_t))
                    
                    l2_errors.append(l2_err)
                    mae_errors.append(mae_err)

            pool_l2_m = np.mean(l2_errors)
            pool_l2_s = np.std(l2_errors)
            pool_mae_m = np.mean(mae_errors)
            pool_mae_s = np.std(mae_errors)

            print(f"\n--- Pooled Error Statistics (n = {test_hearts * num_sims}) ---")
            print(f"Relative L2 Error : {pool_l2_m:.4f} ± {pool_l2_s:.4f}")
            print(f"Mean Absolute Error: {pool_mae_m:.4f} ± {pool_mae_s:.4f} ms")
            print("--------------------------------------------------\n")

            print("Evaluation complete.")

if __name__ == "__main__":
    main()