"""
Independent Evaluation Script for DIMON
Relative L2 and MAE Error Distribution Joyplot
"""

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ParseArgument, to_numpy
from opnn import opnn

def main():
    # 1. Configuration (Matching the training setup)
    args = ParseArgument()
    device = args.device
    
    dim_br_geo =  [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr =      [4, 200, 200, 200, 200] 
    
    save_directory = '3_fs_10000ep_0.0005lr'
    model_path = f'CheckPts/model_chkpts_{save_directory}.pt'
    dump_test = f'./Predictions/{save_directory}/Test/'
    os.makedirs(dump_test, exist_ok=True)

    # 2. Load Dataset
    data_path = "./DIMON_training_data_healthy_2.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    dataset = np.load(data_path)
    theta = dataset['theta']              
    pacing = dataset['pacing']            
    u_all = dataset['u_data']
    cobiveco = dataset['cobiveco']
    anisotropy = dataset['reference_coords'][:, 3:]
    trunk_input = cobiveco
    #trunk_input = np.column_stack([cobiveco, anisotropy])  
    x_coords = dataset['reference_coords'][:,:3]

    # Match original train/test split to compute exact normalization stats
    num_total_train = 125 - 20
    num_val_hearts = 10
    num_train_hearts = num_total_train - num_val_hearts
    
    f_train = theta[:num_train_hearts]
    u_train_raw = u_all[:num_train_hearts]
    f_test = theta[num_total_train:]
    u_test_raw = u_all[num_total_train:]
    
    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)
    x_tensor = torch.tensor(trunk_input, dtype=torch.float).to(device)

    # 3. Normalization (Must replicate training stats precisely)
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_test_norm = (f_test - f_mean) / f_std

    u_mean_train = u_train_raw.min()
    u_std = u_train_raw.std() 

    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)

    # 4. Load Model
    print(f"Loading model weights from {model_path}...")
    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Run Inference
    print("Running inference on the test dataset...")
    with torch.no_grad():
        y_pred_test_raw = model.forward(f_test_tensor, x_pace_tensor, x_tensor)
        u_test_pred = to_numpy(y_pred_test_raw) * u_std + u_mean_train
        u_test_phy = u_test_raw

    # 6. Calculate L2 and MAE Errors
    print("Calculating Errors...")
    num_test_hearts = u_test_pred.shape[0]
    num_sims = 9
    records = []

    for i in range(num_test_hearts):
        for s in range(num_sims):
            u_p = u_test_pred[i, s]
            u_t = u_test_phy[i, s]
            
            # Calculate Relative L2 norm and MAE
            l2_err = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
            mae_err = np.mean(np.abs(u_p - u_t))
            
            records.append({'Pacing_ID': s, 'Relative L2 Error': l2_err, 'MAE': mae_err})
            
    df = pd.DataFrame(records)
    
    # Melt dataframe for side-by-side plotting
    df_melted = df.melt(id_vars=['Pacing_ID'], value_vars=['Relative L2 Error', 'MAE'],
                        var_name='Metric', value_name='Error')

   # 7. Generate Combined Ridge Plot
    print("Generating distribution plot...")
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette("OrRd_r", num_sims)
    
    # ADDED: sharey=False to prevent squashing the density curves
    g = sns.FacetGrid(df_melted, row="Pacing_ID", col="Metric", hue="Pacing_ID", 
                      aspect=5, height=0.6, palette=pal, sharex="col", sharey=False)
    
    g.map(sns.kdeplot, "Error", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.9, linewidth=1.5)
    g.map(sns.kdeplot, "Error", clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Vertical black mean line
    def draw_mean(*args, **kwargs):
        data = kwargs.pop('data')
        mean_val = data['Error'].mean()
        plt.axvline(mean_val, color='k', linestyle='-', lw=1.5)
        
    g.map_dataframe(draw_mean)

    # Formatting axes and overlaps
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.figure.subplots_adjust(hspace=-0.4, wspace=0.1)

    # Add Pacing ID to the left-most axes
    for i, ax in enumerate(g.axes[:, 0]):
        ax.text(-0.02, 0.2, str(i), fontweight="bold", color='black',
                ha="right", va="center", transform=ax.transAxes)

    # Add Column Titles
    g.axes[0, 0].set_title("Relative L2 Error", fontsize=12, pad=20)
    g.axes[0, 1].set_title("Mean Absolute Error (MAE)", fontsize=12, pad=20)

    # Set independent X-axis labels
    g.axes[-1, 0].set_xlabel("Error Value")
    g.axes[-1, 1].set_xlabel("Error Value (ms)")

    plt.suptitle(f"Error Distributions (n = {num_test_hearts})", y=1.05, fontsize=14)
    
    # Save Plot
    plot_path = os.path.join(dump_test, "combined_error_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation complete. Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()