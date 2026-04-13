"""
Dongcheng Cang, dccang@u.nus.edu
Final Integrated DIMON Implementation: Mini-Batching + 3D Evaluation
"""

import os
import torch
import time
import numpy as np
import scipy.io as io
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from opnn import *

import matplotlib.pyplot as plt

def main():
    ## hyperparameters
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    save_step = args.save_step
    test_model = args.test_model
    if test_model == 1:
        load_path = "./CheckPts/model_chkpts.pt"

    # --- Configuration ---
    num_geomode = 5 
    if_normal = 1
    batch_size = 5  
    u_scale = 1.0  

    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr = [3, 200, 200, 200, 200] 

    num_train = 85
    num_test = 20
    
    ## create folders
    dump_test = './Predictions/Test/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## 1. Load Dataset
    dataset = np.load("../DIMON_training_data_healthy.npz")
    theta_all = dataset['theta']              
    pacing_all = dataset['pacing']            
    u_all = dataset['activation_times']       
    x_data = dataset['reference_coords']      

    u_train = u_all[:num_train, :]            
    f_train = theta_all[:num_train, :]     
    x_pace_train = pacing_all[:num_train, :]  

    u_test = u_all[num_train:, :]             
    f_test = theta_all[num_train:, :]      
    x_pace_test = pacing_all[num_train:, :]    

    num_pts = x_data.shape[0]
    u_test_reshape_phy = u_test.reshape(-1, num_pts)

    ## 2. Normalization
    u_test_phy = u_test.copy()
    if if_normal == 1:
        f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
        f_train = (f_train - f_mean) / f_std
        f_test = (f_test - f_mean) / f_std

        u_mean_train = u_train.min()
        u_std = 1.0 
        u_train = (u_train - u_mean_train) / u_std
        u_test = (u_test - u_mean_train) / u_std

    ## 3. Tensors & Loader
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_train_tensor = torch.tensor(f_train, dtype=torch.float).to(device)
    x_pace_train_tensor = torch.tensor(x_pace_train, dtype=torch.float).to(device)
    
    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    x_pace_test_tensor = torch.tensor(x_pace_test, dtype=torch.float).to(device)
    x_tensor = torch.tensor(x_data, dtype=torch.float).to(device)

    train_ds = TensorDataset(f_train_tensor, x_pace_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    if test_model == 0:
        train_loss = np.zeros((args.epochs, 1))
        test_loss = np.zeros((args.epochs, 1))
        mean_abs_err = np.zeros((args.epochs, 1))
        # rel_MSE_err = np.zeros((args.epochs, 1))

        ## 4. Training
        def train_epoch(epoch):
            model.train()
            epoch_loss = 0
            for b_f, b_pace, b_u in train_loader:
                def closure():
                    optimizer.zero_grad()
                    y_pred = model.forward(b_f, b_pace, x_tensor)
                    loss = ((y_pred - b_u)**2).mean()
                    loss.backward()
                    return loss
                l = optimizer.step(closure)
                epoch_loss += l.item()
            train_loss[epoch, 0] = epoch_loss / len(train_loader)

        print('Start training...', flush=True)
        tic = time.time()
        for epoch in range(0, epochs):
            if epoch == 20000:
                for g in optimizer.param_groups: g['lr'] = 0.0001

            train_epoch(epoch)

            ## Testing/Evaluation
            model.eval()
            with torch.no_grad():
                y_pred_raw = model.forward(f_test_tensor, x_pace_test_tensor, x_tensor)
                y_pred = to_numpy(y_pred_raw)
                
                loss_tmp = ((y_pred - u_test)**2).mean()
                test_loss[epoch, 0] = loss_tmp

                ## Physical Error Metrics
                y_pred_phy = y_pred * u_std + u_mean_train
                mean_abs_err[epoch] = (abs(y_pred_phy - u_test_phy) * u_scale).mean()
                
                # rel_MSE_err[epoch] = np.mean(
                #     np.linalg.norm(u_test_reshape_phy - y_pred_phy.reshape(-1, num_pts), axis=1) / 
                #     np.linalg.norm(u_test_reshape_phy, axis=1)
                # )

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Train: {train_loss[epoch, 0]:.6f}, Test: {test_loss[epoch, 0]:.6f}') # Rel L2: {rel_MSE_err[epoch, 0]:.6f}', flush=True
                
            if (epoch+1) % save_step == 0: 
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_path)

        toc = time.time()
        print(f'Total training time: {int((toc-tic)/60)} min', flush=True)
        np.savetxt('./train_loss.txt', train_loss)
        np.savetxt('./test_loss.txt', test_loss)
        # np.savetxt('./rel_MSE_err.txt', rel_MSE_err)

    else:
        # --- 5. Evaluation for both Training and Test Sets ---
        checkpoint = torch.load(load_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("Generating 3D point cloud visualizations...")
            
        # Define number of cases to visualize
        num_viz = min(5, num_test)
            
        with torch.no_grad():
            # Get Predictions for Test Set
            y_pred_test_raw = model.forward(f_test_tensor, x_pace_test_tensor, x_tensor)
            u_test_pred = to_numpy(y_pred_test_raw) * u_std + u_mean_train
                
            # Get Predictions for a subset of the Training Set (first 5 cases)
            f_train_subset = f_train_tensor[:num_viz]
            x_pace_train_subset = x_pace_train_tensor[:num_viz]
            y_pred_train_raw = model.forward(f_train_subset, x_pace_train_subset, x_tensor)
            u_train_pred = to_numpy(y_pred_train_raw) * u_std + u_mean_train
                
            # Ground truth for Training (un-normalized)
            # Based on your script, u_all contains the original physical values
            u_train_phy = u_all[:num_viz, :]

            # Helper function to avoid code duplication
            def plot_comparison(pred_batch, true_batch, folder_name, file_prefix):
                os.makedirs(folder_name, exist_ok=True)
                for i in range(num_viz):
                    u_p = pred_batch[i]
                    u_t = true_batch[i]
                    
                    # Sync scales: find common min/max for the specific case
                    v_min = min(u_t.min(), u_p.min())
                    v_max = max(u_t.max(), u_p.max())
                    
                    fig = plt.figure(figsize=(18, 6))
                    
                    # Panel 1: Ground Truth
                    ax1 = fig.add_subplot(131, projection='3d')
                    sc1 = ax1.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=u_t, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                    ax1.set_title(f"{file_prefix.capitalize()} True (Case {i})")
                    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

                    # Panel 2: DIMON Prediction
                    ax2 = fig.add_subplot(132, projection='3d')
                    sc2 = ax2.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=u_p, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                    ax2.set_title(f"{file_prefix.capitalize()} Pred (Case {i})")
                    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

                    # Panel 3: Absolute Error
                    ax3 = fig.add_subplot(133, projection='3d')
                    err = np.abs(u_t - u_p)
                    sc3 = ax3.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=err, cmap='Reds', s=1, vmin=0)
                    ax3.set_title(f"Abs Error (Mean: {np.mean(err):.4f})")
                    plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

                    for ax in [ax1, ax2, ax3]:
                        ax.set_axis_off()
                        
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_name, f"{file_prefix}_case_{i}_3d.png"), dpi=200)
                    plt.close()

            # Generate plots for both sets
            print(f"Saving training plots to ./Predictions/Train/")
            plot_comparison(u_train_pred, u_train_phy, './Predictions/Train/', 'train')

            print(f"Saving test plots to ./Predictions/Test/")
            plot_comparison(u_test_pred, u_test_phy, './Predictions/Test/', 'test')

            print("Evaluation complete.")

if __name__ == "__main__":
    main()