"""
Dongcheng Cang, dccang@u.nus.edu
Final Integrated DIMON Implementation: 3D Stacked Logic
Mesh Nodes: 48,287 | Hearts: 21 | Paces: 5
"""

import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from opnn import *
import matplotlib.pyplot as plt

def main():
    ## 1. Hyperparameters & Configuration
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    save_step = args.save_step
    test_model = args.test_model
    
    dim_br_geo = [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr = [3, 200, 200, 200, 200] 
    
    dump_test = './Predictions/2/Test/'
    dump_train = './Predictions/2/Train/'
    model_path = 'CheckPts/model_chkpts_2.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## 2. Load Stacked Dataset
    data_path = "./DIMON_training_data_healthy.npz"

    num_train_hearts = 124-20
    batch_size = 10 

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Stacked data not found at {data_path}")
        
    dataset = np.load(data_path)
    theta = dataset['theta']              
    pacing = dataset['pacing']            
    u_all = dataset['u_data']   
    x_coords = dataset['reference_coords']

    f_train = theta[:num_train_hearts]
    u_train_raw = u_all[:num_train_hearts]
    
    f_test = theta[num_train_hearts:]
    u_test_raw = u_all[num_train_hearts:]
    
    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)
    x_tensor = torch.tensor(x_coords, dtype=torch.float).to(device)

    ## 3. Normalization
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_train_norm = (f_train - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

    u_mean_train = u_train_raw.min()
    u_std = u_train_raw.std() 
    u_train_norm = (u_train_raw - u_mean_train) / u_std
    u_test_norm = (u_test_raw - u_mean_train) / u_std

    ## 4. Tensors & DataLoader
    f_train_tensor = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    u_train_tensor = torch.tensor(u_train_norm, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    if test_model == 0:
        print(f"--- Starting Training (Batch Size: {batch_size} hearts) ---")
        train_ds = TensorDataset(f_train_tensor, u_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        train_loss_his = []
        test_loss_his = []
        
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

            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    y_val = model.forward(f_test_tensor, x_pace_tensor, x_tensor)
                    val_loss = ((y_val - torch.tensor(u_test_norm).to(device))**2).mean().item()
                    test_loss_his.append(val_loss)
                    print(f'Epoch: {epoch} | Train Loss: {avg_train_loss:.6f} | Test Loss: {val_loss:.6f}')
            
            # Ensure model is saved periodically
            if (epoch+1) % save_step == 0:
                torch.save({'model_state_dict': model.state_dict()}, model_path)
                print(f"Checkpoint saved to {model_path}")
        
        # Final Save
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        print(f"Final model saved to {model_path}")
        
        print(f"Total training time: {int((time.time()-tic)/60)} min")
        np.savetxt('./train_loss.txt', np.array(train_loss_his))
        np.savetxt('./test_loss.txt', np.array(test_loss_his))

    else:
        # --- 5. Evaluation for both Training and Test Sets ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Generating 3D Evaluation Visualizations (All Pacing Sites) ---")
        num_viz_hearts = 2  # As requested: first 2 test cases
        num_sims = 5        # All simulations by pacing
            
        with torch.no_grad():
            # Get Predictions for Test Set
            y_pred_test_raw = model.forward(f_test_tensor, x_pace_tensor, x_tensor)
            u_test_pred = to_numpy(y_pred_test_raw) * u_std + u_mean_train
                
            # Get Predictions for Training Set subset
            f_train_subset = f_train_tensor[:num_viz_hearts]
            y_pred_train_raw = model.forward(f_train_subset, x_pace_tensor, x_tensor)
            u_train_pred = to_numpy(y_pred_train_raw) * u_std + u_mean_train
                
            u_train_phy = u_train_raw[:num_viz_hearts]
            u_test_phy = u_test_raw[:num_viz_hearts]

            def plot_full_comparison(pred_batch, true_batch, folder_name, file_prefix):
                os.makedirs(folder_name, exist_ok=True)
                # Loop through each heart case
                for i in range(num_viz_hearts):
                    # Loop through all 5 pacing simulations for that heart
                    for s in range(num_sims):
                        u_p = pred_batch[i, s] 
                        u_t = true_batch[i, s]
                        
                        # Sync scales: find common min/max for the specific simulation
                        v_min, v_max = min(u_t.min(), u_p.min()), max(u_t.max(), u_p.max())
                        
                        fig = plt.figure(figsize=(18, 6))
                        
                        # Panel 1: Ground Truth (Eikonal)
                        ax1 = fig.add_subplot(131, projection='3d')
                        sc1 = ax1.scatter(x_coords[:, 0], x_coords[:, 1], x_coords[:, 2], 
                                          c=u_t, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax1.set_title(f"{file_prefix.capitalize()} Ground Truth (Heart {i}, Sim {s})")
                        plt.colorbar(sc1, ax=ax1, shrink=0.5)

                        # Panel 2: DIMON Prediction
                        ax2 = fig.add_subplot(132, projection='3d')
                        sc2 = ax2.scatter(x_coords[:, 0], x_coords[:, 1], x_coords[:, 2], 
                                          c=u_p, cmap='jet', s=1, vmin=v_min, vmax=v_max)
                        ax2.set_title(f"{file_prefix.capitalize()} Pred (Heart {i}, Sim {s})")
                        plt.colorbar(sc2, ax=ax2, shrink=0.5)

                        # Panel 3: Absolute Error (ms)
                        ax3 = fig.add_subplot(133, projection='3d')
                        err = np.abs(u_t - u_p)
                        sc3 = ax3.scatter(x_coords[:, 0], x_coords[:, 1], x_coords[:, 2], 
                                          c=err, cmap='Reds', s=1)
                        ax3.set_title(f"Abs Error (Mean: {np.mean(err):.2f} ms)")
                        plt.colorbar(sc3, ax=ax3, shrink=0.5)

                        for ax in [ax1, ax2, ax3]: ax.set_axis_off()
                        plt.tight_layout()
                        
                        # Save identifying both the heart and the pacing site
                        plt.savefig(os.path.join(folder_name, f"{file_prefix}_heart{i}_sim{s}_batch2.png"), dpi=200)
                        plt.close()

            print(f"Saving comparison plots to {dump_train} and {dump_test}")
            plot_full_comparison(u_train_pred, u_train_phy, dump_train, 'train')
            plot_full_comparison(u_test_pred, u_test_phy, dump_test, 'test')
            print("Multi-simulation evaluation complete.")

if __name__ == "__main__":
    main()