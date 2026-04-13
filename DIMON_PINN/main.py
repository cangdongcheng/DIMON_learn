import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from opnn import *
import random

# 1. Deterministic Seeding to remove the "luck" factor
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(42)

def main():
    ## 1. Hyperparameters & Configuration
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    
    normalize = 1  # 1 to enable normalization, 0 to disable
    set_alpha = 0.05
    
    dim_br_geo =  [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr =      [3, 200, 200, 200, 200]

    batch_size = 24
    learning_rate = 0.0005
    
    save_directory = f'pinn_v600_fixed_alpha{set_alpha}_{epochs}ep_{learning_rate}lr'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## 2. Load Stacked Dataset
    data_path = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV/DIMON_training_data_healthy_fixed.npz"
    data_path2 = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV/reference_cobiveco.npz"

    num_train_hearts = 95
    num_val_hearts = 5
    num_test_hearts = 25
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Stacked data not found at {data_path}")
        
    dataset = np.load(data_path)
    dataset2 = np.load(data_path2)
    theta = dataset['theta']   
    pacing = dataset['pacing'] 
    u_all = dataset['u_data']  
    abrttm = dataset2['cobiveco'][:,:3]
    rvlv = dataset2['cobiveco'][:,4]
    cobiveco = np.column_stack([abrttm,rvlv])
    anisotropy = dataset['ref_anisotropy'] 
    cartesian_coords = dataset['cartesian_coords'] 

    trunk_input = cartesian_coords

    f_train = theta[:num_train_hearts]
    u_train_raw = u_all[:num_train_hearts]
    
    f_val = theta[num_train_hearts:num_train_hearts+num_val_hearts]
    u_val_raw = u_all[num_train_hearts:num_train_hearts+num_val_hearts]
    
    f_test = theta[num_train_hearts+num_val_hearts:num_train_hearts+num_val_hearts+num_test_hearts]
    u_test_raw = u_all[num_train_hearts+num_val_hearts:num_train_hearts+num_val_hearts+num_test_hearts]
    
    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)
    x_tensor = torch.tensor(trunk_input, dtype=torch.float).to(device)

    # Calculate spatial bounds
    x_min = x_tensor.min(dim=0, keepdim=True)[0]
    x_max = x_tensor.max(dim=0, keepdim=True)[0]

    ## 3. Normalization Switch
    if normalize == 1:
        spatial_range = (x_max - x_min).to(device)
        x_tensor_norm = (x_tensor - x_min) / spatial_range
        
        f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
        f_train_norm = (f_train - f_mean) / f_std
        f_val_norm = (f_val - f_mean) / f_std
        f_test_norm = (f_test - f_mean) / f_std

        u_mean_train = u_train_raw.min()
        u_std = u_train_raw.std() 
        u_train_norm = (u_train_raw - u_mean_train) / u_std
        u_val_norm = (u_val_raw - u_mean_train) / u_std
        u_test_norm = (u_test_raw - u_mean_train) / u_std
    else:
        spatial_range = 1.0
        x_tensor_norm = x_tensor
        
        f_mean, f_std = 0.0, 1.0
        f_train_norm = f_train
        f_val_norm = f_val
        f_test_norm = f_test

        u_mean_train = 0.0
        u_std = 1.0
        u_train_norm = u_train_raw
        u_val_norm = u_val_raw
        u_test_norm = u_test_raw

    ## 4. Tensors & DataLoader
    f_train_tensor = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    u_train_tensor = torch.tensor(u_train_norm, dtype=torch.float).to(device)
    f_val_tensor = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    u_val_tensor = torch.tensor(u_val_norm, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)
    u_test_tensor = torch.tensor(u_test_norm, dtype=torch.float).to(device)

    anisotropy_tensor = torch.tensor(anisotropy, dtype=torch.float).to(device)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=400, factor=0.5)

    if test_model == 0:
        print(f"--- Starting Training (Batch Size: {batch_size} hearts) ---")
        train_ds = TensorDataset(f_train_tensor, u_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        train_loss_his = []
        test_loss_his = []  
        best_val_loss = float('inf')
        
        tic = time.time()

        x_tensor_norm.requires_grad_(True)
        
        v_f, v_s, v_n = 600.0, 240.0, 240.0
        f_dir = anisotropy_tensor[:, 0:3]
        s_dir = anisotropy_tensor[:, 3:6]
        n_dir = anisotropy_tensor[:, 6:9]
        
        D_physical = (v_f**2) * torch.einsum('ni,nj->nij', f_dir, f_dir) + \
                     (v_s**2) * torch.einsum('ni,nj->nij', s_dir, s_dir) + \
                     (v_n**2) * torch.einsum('ni,nj->nij', n_dir, n_dir)
        D_physical = D_physical.to(device)

        for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                epoch_data_loss = 0
                epoch_pde_loss = 0  
                
                for b_f, b_u in train_loader:
                    optimizer.zero_grad()
                    
                    y_pred = model.forward(b_f, x_pace_tensor, x_tensor_norm)
                    
                    loss_data = ((y_pred - b_u)**2).mean() 
                    
                    b_idx = torch.randint(0, b_f.size(0), (1,)).item()
                    p_idx = torch.randint(0, b_u.size(1), (1,)).item()
                    
                    y_single = y_pred[b_idx, p_idx, :] 
                    y_pred_physical = y_single * float(u_std) + float(u_mean_train)
                    
                    grad_x_norm = torch.autograd.grad(
                        outputs=y_pred_physical, 
                        inputs=x_tensor_norm, 
                        grad_outputs=torch.ones_like(y_pred_physical), 
                        create_graph=True
                    )[0] 
                    
                    grad_x_physical = grad_x_norm / spatial_range
                    
                    grad_D_grad = torch.einsum('ni,nij,nj->n', grad_x_physical, D_physical, grad_x_physical)
                    residual = torch.sqrt(torch.relu(grad_D_grad) + 1e-8) - 1.0
                    loss_pde = (residual**2).mean()
                    
                    pde_val = loss_pde.detach()
                    data_val = loss_data.detach()
                    
                    alpha = set_alpha
                    
                    if pde_val > 0:
                        lambda_pde = (data_val / pde_val) * alpha
                    else:
                        lambda_pde = 0.0
                    
                    loss = loss_data + lambda_pde * loss_pde
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_data_loss += loss_data.item()
                    epoch_pde_loss += loss_pde.item()  
                
                avg_train_loss = epoch_loss / len(train_loader)
                avg_data_loss = epoch_data_loss / len(train_loader)
                avg_pde_loss = epoch_pde_loss / len(train_loader)  
                train_loss_his.append(avg_train_loss)

                model.eval()
                with torch.no_grad():
                    y_val = model.forward(f_val_tensor, x_pace_tensor, x_tensor_norm)
                    val_loss = ((y_val - u_val_tensor)**2).mean().item()
                    
                    y_test = model.forward(f_test_tensor, x_pace_tensor, x_tensor_norm)
                    test_loss = ((y_test - u_test_tensor)**2).mean().item()
                    test_loss_his.append(test_loss)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({'model_state_dict': model.state_dict()}, model_path)

                if epoch % 100 == 0:
                    print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.6f} | Data Loss: {avg_data_loss:.6f} | PDE Loss: {avg_pde_loss:.6f} | Best Val: {best_val_loss:.6f} | Test Loss: {test_loss:.6f}')        
            
        print(f"Total training time: {int((time.time()-tic)/60)} min")
        np.savetxt(f'./Predictions/{save_directory}/train_loss.txt', np.array(train_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/test_loss.txt', np.array(test_loss_his))

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss_his, label='Train')
        ax.semilogy(test_loss_his, label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(save_directory)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./Predictions/{save_directory}/loss_curve.png', dpi=200)
        plt.close()
        print(f"Loss curve saved to ./Predictions/{save_directory}/loss_curve.png")

    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Measuring Inference Speed ---")
        with torch.no_grad():
            _ = model.forward(f_test_tensor[:1], x_pace_tensor, x_tensor_norm)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        num_test_hearts = f_test_tensor.shape[0]
        num_pacing_sites = x_pace_tensor.shape[0]
        
        start_time = time.time()
        with torch.no_grad():
            y_pred_test_raw = model.forward(f_test_tensor, x_pace_tensor, x_tensor_norm)
            if device == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_heart = total_time / num_test_hearts
        avg_time_per_sim = total_time / (num_test_hearts * num_pacing_sites)
        
        print(f"Total Inference Time ({num_test_hearts} hearts): {total_time:.4f} s")
        print(f"Avg Time per Patient (9 simulations): {avg_time_per_heart*1000:.2f} ms")
        print(f"Avg Time per Individual Simulation: {avg_time_per_sim*1000:.2f} ms")
        print("---------------------------------------\n")

        print("--- Generating 3D Evaluation Visualizations (All Pacing Sites) ---")
        num_viz_hearts = 2 
        num_sims = 3        
            
        with torch.no_grad():
            f_train_subset = f_train_tensor[:num_viz_hearts]

            u_test_pred = to_numpy(y_pred_test_raw) * u_std + u_mean_train
                
            y_pred_train_raw = model.forward(f_train_subset, x_pace_tensor, x_tensor_norm)
            u_train_pred = to_numpy(y_pred_train_raw) * u_std + u_mean_train
                
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
            print("Multi-simulation 3D evaluation complete.")

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

if __name__ == '__main__':
    main()