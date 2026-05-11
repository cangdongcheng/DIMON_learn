"""
Geo-DeepONet: geometry → activation time map on the canonical reference.
Adapted from Cobiveco_with_fiber/main.py with pacing branch removed.

Single fixed 7-node Durrer activation — geometry is the only input variable.

Input:  theta (N, 60) PCA geometry + Cobiveco (M, 4) trunk
Output: AT(x) on reference mesh (N, M)

Data:   geo_deeponet_data.npz from package_training_data.py

Evaluation outputs (--test-model 1) — all SVG, transparent background:
  Test/  heart{h}_AT_{GT,Pred,AbsErr}.svg   — 3D AT scatters (rasterised)
         colorbars/cbar_{AT,AT_AbsErr}.svg   — standalone shared colorbars
         test_predictions.npz                — pred + true + per-case metrics
  Train/ heart{h}_AT_{GT,Pred,AbsErr}.svg   — same, on first --viz-hearts train cases

Shared AT color scale across all viz hearts (train + test). 3D scatters are
rasterised inside the SVG at 300 dpi and rendered in parallel across up to
8 processes.

Eval-time flags:
    --viz-hearts N    number of hearts to plot from train & test (default 2)
    --skip-plots      skip scatter rendering (metrics only)
    --ckpt-path PATH  override checkpoint path (default: derived from args)

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DeepONet

    # Train
    python main.py --epochs 50000 --device cuda

    # Evaluate saved model (matches save_directory → checkpoint)
    python main.py --test-model 1 --device cuda --epochs 10000

    # Explicit checkpoint + skip plots (metrics only)
    python main.py --test-model 1 --device cuda --epochs 10000 \
        --ckpt-path CheckPts/model_chkpts_cobiveco_1norm_10000ep_0.0005lr.pt \
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

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"


# ── Module-level worker for parallel 3D scatter rendering ──────────────────
# ProcessPoolExecutor requires a top-level (picklable) function. The shared
# cartesian_coords is passed via the pool initializer to avoid re-serializing
# it for every task.
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
    set_seed(42)

    ## 1. Hyperparameters & Configuration
    args = ParseArgument()
    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    sync = (lambda: torch.cuda.synchronize()) if 'cuda' in str(device) else (lambda: None)

    normalize = 1

    # Architecture dimensions
    num_geomode = 60
    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_tr = [4, 200, 200, 200, 200]  # 4D Cobiveco trunk

    batch_size = 96
    learning_rate = 0.0005

    save_directory = f'cobiveco_{normalize}norm_{epochs}ep_{learning_rate}lr'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = args.ckpt_path or f'CheckPts/model_chkpts_{save_directory}.pt'
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
        ax.semilogy(test_loss_his, label='Test', alpha=0.8)
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
        num_viz_hearts = min(args.viz_hearts, num_test_hearts)

        # Load cartesian coords for 3D plotting (shared with worker pool)
        dimon_data = np.load(os.path.join(os.path.dirname(__file__), "..",
                                           "DIMON_training_data_healthy.npz"))
        cartesian_coords = dimon_data['cartesian_coords']  # (50797, 3)

        with torch.no_grad():
            # Warm up (first call has overhead)
            _ = model.forward(f_test_tensor[:1], x_tensor)
            sync()

            # Per-case inference timing
            infer_times = []
            all_pred = []
            for i in range(num_test_hearts):
                sync()
                t0 = time.perf_counter()
                y = model.forward(f_test_tensor[i:i + 1], x_tensor)
                sync()
                infer_times.append(time.perf_counter() - t0)
                all_pred.append(to_numpy(y[0]))
            y_test_norm = np.stack(all_pred)
            u_test_pred = y_test_norm * u_std_train + u_mean_train

            print(f"Inference: {np.mean(infer_times)*1000:.1f} +/- "
                  f"{np.std(infer_times)*1000:.1f} ms/case "
                  f"(total {np.sum(infer_times):.2f} s for {num_test_hearts} cases)")

            # Training-set viz predictions (small)
            f_train_subset = f_train_tensor[:num_viz_hearts]
            y_train_norm = model.forward(f_train_subset, x_tensor)
            u_train_pred = to_numpy(y_train_norm) * u_std_train + u_mean_train
            u_train_phy = u_train_raw[:num_viz_hearts]

        # --- Error Statistics ---
        print(f"\n{'Case':<35} {'Rel L2':>10} {'MAE (ms)':>10}")
        print("-" * 58)
        l2_errors, mae_errors = [], []
        for i in range(num_test_hearts):
            u_p = u_test_pred[i]
            u_t = u_test_raw[i]
            l2_err = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
            mae_err = np.mean(np.abs(u_p - u_t))
            l2_errors.append(l2_err)
            mae_errors.append(mae_err)
            print(f"{case_names[num_train_hearts + num_val_hearts + i]:<35}"
                  f" {l2_err:10.4f} {mae_err:10.2f}")
        print("-" * 58)
        print(f"Rel L2 = {np.mean(l2_errors):.4f} ± {np.std(l2_errors):.4f} | "
              f"MAE = {np.mean(mae_errors):.2f} ± {np.std(mae_errors):.2f} ms")

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
            print("Skipping 3D scatter rendering (--skip-plots)")
        else:
            # Shared AT color scale across viz hearts (both train + test)
            gt_stack = np.concatenate([
                u_test_raw[:num_viz_hearts].ravel(),
                u_train_phy.ravel(),
            ])
            pr_stack = np.concatenate([
                u_test_pred[:num_viz_hearts].ravel(),
                u_train_pred.ravel(),
            ])
            at_vmin = float(min(gt_stack.min(), pr_stack.min()))
            at_vmax = float(max(gt_stack.max(), pr_stack.max()))

            err_stack = np.concatenate([
                np.abs(u_test_raw[:num_viz_hearts] - u_test_pred[:num_viz_hearts]).ravel(),
                np.abs(u_train_phy - u_train_pred).ravel(),
            ])
            err_vmax = float(err_stack.max())

            cmap_at = 'RdYlBu_r'
            cmap_err = 'Reds'

            plot_tasks = []  # (values, cmap, vmin, vmax, out_path)

            def queue(values, cmap, vmin, vmax, out_path):
                plot_tasks.append((np.asarray(values, dtype=np.float32),
                                   cmap, float(vmin), float(vmax), out_path))

            for i in range(num_viz_hearts):
                # Test
                err_t = np.abs(u_test_raw[i] - u_test_pred[i])
                tag = f"heart{i}"
                queue(u_test_raw[i], cmap_at, at_vmin, at_vmax,
                      os.path.join(dump_test, f"{tag}_AT_GT.svg"))
                queue(u_test_pred[i], cmap_at, at_vmin, at_vmax,
                      os.path.join(dump_test, f"{tag}_AT_Pred.svg"))
                queue(err_t, cmap_err, 0.0, err_vmax,
                      os.path.join(dump_test, f"{tag}_AT_AbsErr.svg"))

                # Train
                err_tr = np.abs(u_train_phy[i] - u_train_pred[i])
                queue(u_train_phy[i], cmap_at, at_vmin, at_vmax,
                      os.path.join(dump_train, f"{tag}_AT_GT.svg"))
                queue(u_train_pred[i], cmap_at, at_vmin, at_vmax,
                      os.path.join(dump_train, f"{tag}_AT_Pred.svg"))
                queue(err_tr, cmap_err, 0.0, err_vmax,
                      os.path.join(dump_train, f"{tag}_AT_AbsErr.svg"))

            n_workers = min(8, (os.cpu_count() or 4))
            print(f"Rendering {len(plot_tasks)} SVG scatters on "
                  f"{n_workers} processes ...", flush=True)
            t0_render = time.perf_counter()
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_plot_worker,
                                     initargs=(cartesian_coords,)) as ex:
                list(ex.map(_render_scatter_svg, plot_tasks))
            print(f"  done in {time.perf_counter() - t0_render:.1f} s")

            # Standalone colorbars (shared across all scatters above)
            cbar_dir = os.path.join(dump_test, "colorbars")
            os.makedirs(cbar_dir, exist_ok=True)
            save_colorbar(cmap_at, at_vmin, at_vmax, 'AT (ms)',
                          os.path.join(cbar_dir, 'cbar_AT.svg'))
            save_colorbar(cmap_err, 0.0, err_vmax, '|ΔAT| (ms)',
                          os.path.join(cbar_dir, 'cbar_AT_AbsErr.svg'),
                          half=True)
            print(f"Saved colorbars to {cbar_dir}")

        # Save predictions
        np.savez_compressed(
            os.path.join(dump_test, "test_predictions.npz"),
            pred=u_test_pred, true=u_test_raw,
            l2_errors=np.array(l2_errors), mae_errors=np.array(mae_errors),
            case_names=case_names[num_train_hearts + num_val_hearts:])
        print(f"\nEvaluation complete. Plots in {dump_test}")


if __name__ == "__main__":
    main()
