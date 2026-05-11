"""
Dongcheng Cang, dccang@u.nus.edu
Final Integrated DIMON Implementation: 3D Stacked Logic
Mesh Nodes: 50797 | Hearts: 125 | Paces: 9

3-branch MIONet (geo + pacing + 4D Cobiveco trunk) predicting Eikonal
activation times. Output: (N, 9, M).

Evaluation outputs (--test-model 1) — all SVG, transparent background:
  Test/  heart{h}_sim{s}_AT_{GT,Pred,AbsErr}.svg   — 3D AT scatters (rasterised)
         colorbars/cbar_{AT,AT_AbsErr}.svg          — standalone shared colorbars
         combined_error_distribution.svg            — KDE over all test hearts
         test_predictions.npz                       — pred + true + L2/MAE matrices
  Train/ heart{h}_sim{s}_AT_{GT,Pred,AbsErr}.svg   — same, on first --viz-hearts train cases

Shared AT color scale across all viz (heart, sim) pairs. 3D scatters are
rasterised inside the SVG at 300 dpi and rendered in parallel across up to
8 processes. Per-pacing and pooled metrics printed in mean ± std format.

Eval-time flags:
    --viz-hearts N    hearts to plot from train & test (default 2)
    --viz-sims N      pacing sims per heart (default 3, max 9)
    --skip-plots      skip scatter + KDE rendering (metrics only)
    --ckpt-path PATH  override checkpoint path (default: derived from args)

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Cobiveco_with_fiber

    # Train
    python main.py --epochs 50000 --device cuda

    # Evaluate saved model (matches save_directory → checkpoint)
    python main.py --test-model 1 --device cuda --epochs 10000

    # Metrics only (skip plots), explicit checkpoint
    python main.py --test-model 1 --device cuda --epochs 10000 \
        --ckpt-path CheckPts/model_chkpts_cobiveco4d_1norm_10000ep_0.0005lr.pt \
        --skip-plots
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
from concurrent.futures import ProcessPoolExecutor


# ── Module-level worker for parallel 3D scatter rendering ──────────────────
_WORKER_XYZ = None


def _init_plot_worker(xyz):
    global _WORKER_XYZ
    _WORKER_XYZ = xyz
    import matplotlib
    matplotlib.use('Agg')


def _render_scatter_svg(task):
    values, cmap, vmin, vmax, out_path = task
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)
    ax.scatter(_WORKER_XYZ[:, 0], _WORKER_XYZ[:, 1], _WORKER_XYZ[:, 2],
               c=values, cmap=cmap, s=1,
               vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, format='svg', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close()


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
    sync = (lambda: torch.cuda.synchronize()) if 'cuda' in str(device) else (lambda: None)
    
    normalize = 1  # Switch: 1 to enable, 0 to disable
    
    # Architecture dimensions
    dim_br_geo =  [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200] 
    dim_tr =      [4, 200, 200, 200, 200]

    batch_size = 48
    learning_rate = 0.0005
    
    save_directory = f'cobiveco4d_{normalize}norm_{epochs}ep_{learning_rate}lr'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = args.ckpt_path or f'CheckPts/model_chkpts_{save_directory}.pt'
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
    
    # Prepare Cobiveco (4 channels: ab, rt, tm, rvlv)
    cobiveco_coords = dataset2['cobiveco']
    cobiveco_input = np.column_stack([cobiveco_coords[:, :3], cobiveco_coords[:, 4]])
    
    # Prepare Anisotropy (9 channels: f_xyz, s_xyz, n_xyz)
    anisotropy = dataset['ref_anisotropy']
    cartesian_coords = dataset['cartesian_coords']

    # Trunk Input: 4D Cobiveco (ab, rt, tm, tv)
    trunk_raw = cobiveco_coords[:, :4]

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
        ax.semilogy(test_loss_his, label='Test', alpha=0.8)
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
        # --- Evaluation ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("--- Generating Evaluation ---")
        num_viz_hearts = min(args.viz_hearts, num_test_hearts)
        num_viz_sims = min(args.viz_sims, 9)

        with torch.no_grad():
            # Warm up (first call has overhead)
            _ = model.forward(f_test_tensor[:1], x_pace_tensor, x_tensor)
            sync()

            # Per-heart inference timing
            infer_times = []
            all_pred = []
            for i in range(num_test_hearts):
                sync()
                t0 = time.perf_counter()
                y = model.forward(f_test_tensor[i:i + 1], x_pace_tensor, x_tensor)
                sync()
                infer_times.append(time.perf_counter() - t0)
                all_pred.append(to_numpy(y[0]))  # (9, M)
            u_test_pred = np.stack(all_pred) * u_std_train + u_mean_train

            print(f"Inference: {np.mean(infer_times)*1000:.1f} +/- "
                  f"{np.std(infer_times)*1000:.1f} ms/case "
                  f"(total {np.sum(infer_times):.2f} s for {num_test_hearts} hearts × 9 pacings)")

            # Training-set viz predictions
            f_train_subset = f_train_tensor[:num_viz_hearts]
            y_train = model.forward(f_train_subset, x_pace_tensor, x_tensor)
            u_train_pred = to_numpy(y_train) * u_std_train + u_mean_train
            u_train_phy = u_train_raw[:num_viz_hearts]

        # --- Error statistics (per-pacing + pooled, ± format) ---
        n_sims = 9
        l2_mat = np.zeros((num_test_hearts, n_sims))
        mae_mat = np.zeros((num_test_hearts, n_sims))
        for i in range(num_test_hearts):
            for s in range(n_sims):
                u_p = u_test_pred[i, s]
                u_t = u_test_raw[i, s]
                l2_mat[i, s] = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                mae_mat[i, s] = np.mean(np.abs(u_p - u_t))

        print("\n--- Per-pacing error (mean ± std over test hearts) ---")
        for s in range(n_sims):
            print(f"Pacing {s}: L2 = {l2_mat[:, s].mean():.4f} ± "
                  f"{l2_mat[:, s].std():.4f} | MAE = "
                  f"{mae_mat[:, s].mean():.4f} ± {mae_mat[:, s].std():.4f} ms")

        print("\n--- Pooled ---")
        print(f"Rel L2 = {l2_mat.mean():.4f} ± {l2_mat.std():.4f} | "
              f"MAE = {mae_mat.mean():.2f} ± {mae_mat.std():.2f} ms")

        def save_colorbar(cmap, vmin, vmax, label, out_path,
                          orientation='vertical', half=False):
            if orientation == 'vertical':
                figsize = (1.2, 2.5) if half else (1.2, 5)
            else:
                figsize = (2.5, 1.2) if half else (5, 1.2)
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cb = plt.colorbar(sm, cax=ax, orientation=orientation)
            cb.set_label(label, fontsize=15)
            from matplotlib.ticker import MaxNLocator
            nice = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
            nice = nice[(nice >= vmin) & (nice <= vmax)]
            cb.set_ticks(nice)
            cb.ax.tick_params(labelsize=15)
            plt.tight_layout()
            plt.savefig(out_path, format='svg', bbox_inches='tight',
                        transparent=True)
            plt.close()

        # --- Plotting ---
        if args.skip_plots:
            print("\nSkipping 3D scatter + KDE rendering (--skip-plots)")
        else:
            # Individual SVGs — shared AT color scale across all viz instances
            gt_stack = np.concatenate([
                u_test_raw[:num_viz_hearts, :num_viz_sims].ravel(),
                u_train_phy[:, :num_viz_sims].ravel(),
            ])
            pr_stack = np.concatenate([
                u_test_pred[:num_viz_hearts, :num_viz_sims].ravel(),
                u_train_pred[:, :num_viz_sims].ravel(),
            ])
            at_vmin = float(min(gt_stack.min(), pr_stack.min()))
            at_vmax = float(max(gt_stack.max(), pr_stack.max()))

            err_stack = np.concatenate([
                np.abs(u_test_raw[:num_viz_hearts, :num_viz_sims]
                       - u_test_pred[:num_viz_hearts, :num_viz_sims]).ravel(),
                np.abs(u_train_phy[:, :num_viz_sims]
                       - u_train_pred[:, :num_viz_sims]).ravel(),
            ])
            err_vmax = float(err_stack.max())

            cmap_at = 'RdYlBu_r'
            cmap_err = 'Reds'

            plot_tasks = []

            def queue(values, cmap, vmin, vmax, out_path):
                plot_tasks.append((np.asarray(values, dtype=np.float32),
                                   cmap, float(vmin), float(vmax), out_path))

            for i in range(num_viz_hearts):
                for s in range(num_viz_sims):
                    tag = f"heart{i}_sim{s}"
                    # Test
                    err_t = np.abs(u_test_raw[i, s] - u_test_pred[i, s])
                    queue(u_test_raw[i, s], cmap_at, at_vmin, at_vmax,
                          os.path.join(dump_test, f"{tag}_AT_GT.svg"))
                    queue(u_test_pred[i, s], cmap_at, at_vmin, at_vmax,
                          os.path.join(dump_test, f"{tag}_AT_Pred.svg"))
                    queue(err_t, cmap_err, 0.0, err_vmax,
                          os.path.join(dump_test, f"{tag}_AT_AbsErr.svg"))
                    # Train
                    err_tr = np.abs(u_train_phy[i, s] - u_train_pred[i, s])
                    queue(u_train_phy[i, s], cmap_at, at_vmin, at_vmax,
                          os.path.join(dump_train, f"{tag}_AT_GT.svg"))
                    queue(u_train_pred[i, s], cmap_at, at_vmin, at_vmax,
                          os.path.join(dump_train, f"{tag}_AT_Pred.svg"))
                    queue(err_tr, cmap_err, 0.0, err_vmax,
                          os.path.join(dump_train, f"{tag}_AT_AbsErr.svg"))

            n_workers = min(8, (os.cpu_count() or 4))
            print(f"\nRendering {len(plot_tasks)} SVG scatters on "
                  f"{n_workers} processes ...", flush=True)
            t0_render = time.perf_counter()
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_plot_worker,
                                     initargs=(cartesian_coords,)) as ex:
                list(ex.map(_render_scatter_svg, plot_tasks))
            print(f"  done in {time.perf_counter() - t0_render:.1f} s")

            # Standalone colorbars
            cbar_dir = os.path.join(dump_test, "colorbars")
            os.makedirs(cbar_dir, exist_ok=True)
            save_colorbar(cmap_at, at_vmin, at_vmax, 'AT (ms)',
                          os.path.join(cbar_dir, 'cbar_AT.svg'))
            save_colorbar(cmap_err, 0.0, err_vmax, '|ΔAT| (ms)',
                          os.path.join(cbar_dir, 'cbar_AT_AbsErr.svg'),
                          half=True)
            print(f"Saved colorbars to {cbar_dir}")

            # KDE distribution plot (full test set, all pacings)
            import pandas as pd
            import seaborn as sns
            records = [{'Pacing_ID': s, 'Relative L2 Error': l2_mat[i, s],
                        'MAE': mae_mat[i, s]}
                       for i in range(num_test_hearts) for s in range(n_sims)]
            df = pd.DataFrame(records)
            df_melted = df.melt(id_vars=['Pacing_ID'],
                                value_vars=['Relative L2 Error', 'MAE'],
                                var_name='Metric', value_name='Error')

            print("Generating distribution plot...")
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
            pal = sns.color_palette("OrRd_r", n_sims)
            g = sns.FacetGrid(df_melted, row="Pacing_ID", col="Metric",
                              hue="Pacing_ID", aspect=5, height=0.6,
                              palette=pal, sharex="col", sharey=False)
            g.map(sns.kdeplot, "Error", bw_adjust=0.5, clip_on=False,
                  fill=True, alpha=0.9, linewidth=1.5)
            g.map(sns.kdeplot, "Error", clip_on=False, color="w", lw=2,
                  bw_adjust=0.5)
            g.refline(y=0, linewidth=2, linestyle="-", color=None,
                      clip_on=False)

            def draw_mean(*args, **kwargs):
                data = kwargs.pop('data')
                plt.axvline(data['Error'].mean(), color='k',
                            linestyle='-', lw=1.5)

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
            plt.suptitle(f"Error Distributions (n = {num_test_hearts})",
                         y=1.05, fontsize=14)

            kde_path = os.path.join(dump_test, "combined_error_distribution.svg")
            plt.savefig(kde_path, format='svg', bbox_inches='tight',
                        transparent=True)
            plt.close()
            print(f"Saved KDE plot: {kde_path}")

        # Save predictions
        np.savez_compressed(
            os.path.join(dump_test, "test_predictions.npz"),
            pred=u_test_pred, true=u_test_raw,
            l2=l2_mat, mae=mae_mat)
        print(f"\nEvaluation complete. Outputs in {dump_test}")

if __name__ == "__main__":
    main()