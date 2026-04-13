"""
Dongcheng Cang, dccang@u.nus.edu
Final Integrated DIMON Implementation: 3D Stacked Logic
Mesh Nodes: 50797 | Hearts: 125 | Paces: 9
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

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ## 1. Hyperparameters & Configuration
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    
    normalize = 1  # Switch: 1 to enable, 0 to disable
    
    # Architecture dimensions
    # dim_tr: 13 (4 Cobiveco + 9 Anisotropy)
    dim_br_geo =  [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr =      [4, 200, 200, 200, 200]

    batch_size = 10
    learning_rate = 0.0005
    
    save_directory = f'cobiveco_{normalize}norm_{epochs}ep_{learning_rate}lr'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## 2. Load Stacked Dataset
    data_path = "../DIMON_training_data_healthy.npz"
    data_path2 = "../reference_cobiveco.npz"

    num_train_hearts = 95
    num_val_hearts = 10
    num_test_hearts = 20

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Stacked data not found at {data_path}")
        
    dataset = np.load(data_path)
    dataset2 = np.load(data_path2)
    theta = dataset['theta']              
    pacing = dataset['pacing'] 
    u_all = dataset['u_data']
    
    # Prepare Cobiveco (4 channels: ab, rt, tm, rvlv)
    cobiveco_coords = dataset2['cobiveco']
    cobiveco_input = np.column_stack([cobiveco_coords[:, :3], cobiveco_coords[:, 4]])
    
    # Prepare Anisotropy (9 channels: f_xyz, s_xyz, n_xyz)
    anisotropy = dataset['ref_anisotropy']
    cartesian_coords = dataset['cartesian_coords']

    # Combine into 13-D Trunk Input
    trunk_raw = cobiveco_coords[:,:4]

    f_train = theta[:num_train_hearts]
    u_train_raw = u_all[:num_train_hearts]
    
    f_val = theta[num_train_hearts:num_train_hearts+num_val_hearts]
    u_val_raw = u_all[num_train_hearts:num_train_hearts+num_val_hearts]
    
    f_test = theta[num_train_hearts+num_val_hearts:num_train_hearts+num_val_hearts+num_test_hearts]
    u_test_raw = u_all[num_train_hearts+num_val_hearts:num_train_hearts+num_val_hearts+num_test_hearts]
    
    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)
    x_tensor_raw = torch.tensor(trunk_raw, dtype=torch.float).to(device)

    ## 3. Normalization Logic
    if normalize == 1:
        # Normalize Trunk Input (per-feature min-max to preserve relative bounds)
        x_min = x_tensor_raw.min(dim=0, keepdim=True)[0]
        x_max = x_tensor_raw.max(dim=0, keepdim=True)[0]
        x_tensor = (x_tensor_raw - x_min) / (x_max - x_min + 1e-8)
        
        # Normalize Branch Geometry
        f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
        f_train_norm = (f_train - f_mean) / f_std
        f_val_norm = (f_val - f_mean) / f_std
        f_test_norm = (f_test - f_mean) / f_std

        # Normalize Labels (Using strictly training statistics)
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

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if test_model == 0:
        print(f"--- Starting Training (Batch Size: {batch_size} hearts) ---")
        print(f"Trunk Input Dimension: {trunk_raw.shape[1]}")
        
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
                y_pred = model.forward(b_f, x_pace_tensor, x_tensor)
                loss = ((y_pred - b_u)**2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_loss_his.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                y_val = model.forward(f_val_tensor, x_pace_tensor, x_tensor)
                val_loss = ((y_val - u_val_tensor)**2).mean().item()
                y_test = model.forward(f_test_tensor, x_pace_tensor, x_tensor)
                test_loss = ((y_test - u_test_tensor)**2).mean().item()
                test_loss_his.append(test_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Best Val: {best_val_loss:.6f} | Test Loss: {test_loss:.6f}')
        
        print(f"Total training time: {int((time.time()-tic)/60)} min")
        np.savetxt(f'./Predictions/{save_directory}/train_loss.txt', np.array(train_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/test_loss.txt', np.array(test_loss_his))

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
        # --- 5. Evaluation for both Training and Test Sets ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Generating 3D Evaluation Visualizations (All Pacing Sites) ---")
        num_viz_hearts = 2  # As requested: first 2 test cases
        num_sims = 3        # All simulations by pacing
            
        with torch.no_grad():
            f_train_subset = f_train_tensor[:num_viz_hearts]

            # Denormalization using strictly training stats (u_std_train, u_mean_train)
            y_pred_test_raw = model.forward(f_test_tensor, x_pace_tensor, x_tensor)
            u_test_pred = to_numpy(y_pred_test_raw) * u_std_train + u_mean_train
                
            y_pred_train_raw = model.forward(f_train_subset, x_pace_tensor, x_tensor)
            u_train_pred = to_numpy(y_pred_train_raw) * u_std_train + u_mean_train
                
            u_train_phy = u_train_raw[:num_viz_hearts]
            u_test_phy_viz = u_test_raw[:num_viz_hearts] 

            def plot_full_comparison(pred_batch, true_batch, folder_name, file_prefix):
                os.makedirs(folder_name, exist_ok=True)
                for i in range(num_viz_hearts):
                    for s in range(num_sims):
                        u_p = pred_batch[i, s] 
                        u_t = true_batch[i, s]
                        
                        v_min, v_max = min(u_t.min(), u_p.min()), max(u_t.max(), u_p.max())
                        
                        fig = plt.figure(figsize=(18, 6))
                        
                        ax1 = fig.add_subplot(131, projection='3d')
                        sc1 = ax1.scatter(cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2], 
                                          c=u_t, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax1.set_title(f"{file_prefix.capitalize()} Ground Truth (Heart {i}, Sim {s})")
                        plt.colorbar(sc1, ax=ax1, shrink=0.5)

                        ax2 = fig.add_subplot(132, projection='3d')
                        sc2 = ax2.scatter(cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2], 
                                          c=u_p, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax2.set_title(f"{file_prefix.capitalize()} Pred (Heart {i}, Sim {s})")
                        plt.colorbar(sc2, ax=ax2, shrink=0.5)

                        ax3 = fig.add_subplot(133, projection='3d')
                        err = np.abs(u_t - u_p)
                        sc3 = ax3.scatter(cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2], 
                                          c=err, cmap='Reds', s=1)
                        ax3.set_title(f"Abs Error (Mean: {np.mean(err):.2f} ms)")
                        plt.colorbar(sc3, ax=ax3, shrink=0.5)

                        for ax in [ax1, ax2, ax3]: ax.set_axis_off()
                        plt.tight_layout()
                        
                        plt.savefig(os.path.join(folder_name, f"{file_prefix}_heart{i}_sim{s}.png"), dpi=200)
                        plt.close()

            print(f"Saving comparison plots to {dump_train} and {dump_test}")
            #plot_full_comparison(u_train_pred, u_train_phy, dump_train, 'train')
            #plot_full_comparison(u_test_pred, u_test_phy_viz, dump_test, 'test')
            print("Multi-simulation 3D evaluation complete.")

            # --- 6. Calculate L2 and MAE Errors (Full Test Set) ---
            print("Calculating Errors for Distribution Plot...")
            import pandas as pd
            import seaborn as sns
            
            num_test_hearts = u_test_pred.shape[0]
            u_test_phy_full = u_test_raw 
            records = []

            for i in range(num_test_hearts):
                for s in range(9):
                    u_p = u_test_pred[i, s]
                    u_t = u_test_phy_full[i, s]
                    
                    l2_err = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                    mae_err = np.mean(np.abs(u_p - u_t))
                    
                    records.append({'Pacing_ID': s, 'Relative L2 Error': l2_err, 'MAE': mae_err})
                    
            df = pd.DataFrame(records)
            
            print("\n--- Error Statistics (Mean ± Std) ---")
            stats_df = df.groupby('Pacing_ID').agg({'Relative L2 Error': ['mean', 'std'], 'MAE': ['mean', 'std']})
            
            for s in sorted(df['Pacing_ID'].unique()):
                l2_m = stats_df.loc[s, ('Relative L2 Error', 'mean')]
                l2_s = stats_df.loc[s, ('Relative L2 Error', 'std')]
                mae_m = stats_df.loc[s, ('MAE', 'mean')]
                mae_s = stats_df.loc[s, ('MAE', 'std')]
                print(f"Pacing {s}: L2 = {l2_m:.4f} ± {l2_s:.4f} | MAE = {mae_m:.4f} ± {mae_s:.4f} ms")

            pool_l2_m = df['Relative L2 Error'].mean()
            pool_l2_s = df['Relative L2 Error'].std()
            pool_mae_m = df['MAE'].mean()
            pool_mae_s = df['MAE'].std()
            print(f"Pooled  : L2 = {pool_l2_m:.4f} ± {pool_l2_s:.4f} | MAE = {pool_mae_m:.4f} ± {pool_mae_s:.4f} ms")
            print("---------------------------------------\n")
            
            df_melted = df.melt(id_vars=['Pacing_ID'], value_vars=['Relative L2 Error', 'MAE'],
                                var_name='Metric', value_name='Error')

            print("Generating distribution plot...")
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
            pal = sns.color_palette("OrRd_r", num_sims)
            
            g = sns.FacetGrid(df_melted, row="Pacing_ID", col="Metric", hue="Pacing_ID", 
                              aspect=5, height=0.6, palette=pal, sharex="col", sharey=False)
            
            g.map(sns.kdeplot, "Error", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.9, linewidth=1.5)
            g.map(sns.kdeplot, "Error", clip_on=False, color="w", lw=2, bw_adjust=0.5)
            g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

            def draw_mean(*args, **kwargs):
                data = kwargs.pop('data')
                mean_val = data['Error'].mean()
                plt.axvline(mean_val, color='k', linestyle='-', lw=1.5)
                
            g.map_dataframe(draw_mean)

            g.set_titles("")
            g.set(yticks=[], ylabel="")
            g.despine(bottom=True, left=True)
            g.figure.subplots_adjust(hspace=-0.4, wspace=0.1)

            for i, ax in enumerate(g.axes[:, 0]):
                ax.text(-0.02, 0.2, str(i), fontweight="bold", color='black',
                        ha="right", va="center", transform=ax.transAxes)

            g.axes[0, 0].set_title("Relative L2 Error", fontsize=12, pad=20)
            g.axes[0, 1].set_title("Mean Absolute Error (MAE)", fontsize=12, pad=20)

            g.axes[-1, 0].set_xlabel("Error Value")
            g.axes[-1, 1].set_xlabel("Error Value (ms)")
            g.figure.set_size_inches(12, 8)

            for ax in g.axes[:, 0]:
                ax.set_xlim(0.00, 0.10)
            for ax in g.axes[:, 1]:
                ax.set_xlim(2.0, 10.0)

            plt.suptitle(f"Error Distributions (n = {num_test_hearts})", y=1.05, fontsize=14)
            
            plot_path = os.path.join(dump_test, "combined_error_distribution.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            print(f"Evaluation complete. Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()