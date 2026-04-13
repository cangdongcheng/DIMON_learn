"""
DIMON-PINN 5-fold cross-validation.

Shuffles the 125 hearts with a fixed seed, splits into 5 folds of 25 each.
Each fold: 100 train (95 fit + 5 val for early stopping) / 25 test.
Reports per-fold and pooled Rel L2 and MAE across all 9 pacing sites.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/DIMON_PINN

    python main_cv.py --epochs 2000 --device cuda
    python main_cv.py --epochs 5000 --device cuda
"""
import os
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from opnn import *
import random

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"
N_FOLDS = 5


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(fold_id, train_idx, val_idx, test_idx,
               theta_all, pacing, u_all, trunk_input,
               anisotropy, set_alpha, epochs, device, save_dir):
    """Train and evaluate one fold. Returns per-case (l2, mae) for test set."""

    dim_br_geo  = [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200]
    dim_tr      = [3, 200, 200, 200, 200]
    batch_size = 24
    learning_rate = 0.0005

    # Split
    f_train = theta_all[train_idx]
    u_train_raw = u_all[train_idx]
    f_val = theta_all[val_idx]
    u_val_raw = u_all[val_idx]
    f_test = theta_all[test_idx]
    u_test_raw = u_all[test_idx]

    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)
    x_tensor = torch.tensor(trunk_input, dtype=torch.float).to(device)

    # Spatial normalization
    x_min = x_tensor.min(dim=0, keepdim=True)[0]
    x_max = x_tensor.max(dim=0, keepdim=True)[0]
    spatial_range = (x_max - x_min).to(device)
    x_tensor_norm = (x_tensor - x_min) / spatial_range

    # Geo branch normalization (per-fold, from training set only)
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_norm = (f_train - f_mean) / f_std
    f_val_norm = (f_val - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

    # Output normalization
    u_mean_train = u_train_raw.min()
    u_std_train = u_train_raw.std()
    u_train_norm = (u_train_raw - u_mean_train) / u_std_train
    u_val_norm = (u_val_raw - u_mean_train) / u_std_train

    # Tensors
    f_train_t = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    u_train_t = torch.tensor(u_train_norm, dtype=torch.float).to(device)
    f_val_t = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    u_val_t = torch.tensor(u_val_norm, dtype=torch.float).to(device)
    f_test_t = torch.tensor(f_test_norm, dtype=torch.float).to(device)

    anisotropy_tensor = torch.tensor(anisotropy, dtype=torch.float).to(device)

    # Velocity tensor for PDE loss
    v_f, v_s, v_n = 600.0, 240.0, 240.0
    f_dir = anisotropy_tensor[:, 0:3]
    s_dir = anisotropy_tensor[:, 3:6]
    n_dir = anisotropy_tensor[:, 6:9]
    D_physical = (v_f**2) * torch.einsum('ni,nj->nij', f_dir, f_dir) + \
                 (v_s**2) * torch.einsum('ni,nj->nij', s_dir, s_dir) + \
                 (v_n**2) * torch.einsum('ni,nj->nij', n_dir, n_dir)
    D_physical = D_physical.to(device)

    train_ds = TensorDataset(f_train_t, u_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=400, factor=0.5)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    x_tensor_norm.requires_grad_(True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for b_f, b_u in train_loader:
            optimizer.zero_grad()

            y_pred = model.forward(b_f, x_pace_tensor, x_tensor_norm)
            loss_data = ((y_pred - b_u)**2).mean()

            # PDE residual on one random (heart, pacing) sample
            b_idx = torch.randint(0, b_f.size(0), (1,)).item()
            p_idx = torch.randint(0, b_u.size(1), (1,)).item()

            y_single = y_pred[b_idx, p_idx, :]
            y_pred_physical = y_single * float(u_std_train) + float(u_mean_train)

            grad_x_norm = torch.autograd.grad(
                outputs=y_pred_physical,
                inputs=x_tensor_norm,
                grad_outputs=torch.ones_like(y_pred_physical),
                create_graph=True
            )[0]

            grad_x_physical = grad_x_norm / spatial_range
            grad_D_grad = torch.einsum('ni,nij,nj->n',
                                       grad_x_physical, D_physical, grad_x_physical)
            residual = torch.sqrt(torch.relu(grad_D_grad) + 1e-8) - 1.0
            loss_pde = (residual**2).mean()

            # Adaptive weighting
            pde_val = loss_pde.detach()
            data_val = loss_data.detach()
            if pde_val > 0:
                lambda_pde = (data_val / pde_val) * set_alpha
            else:
                lambda_pde = 0.0

            loss = loss_data + lambda_pde * loss_pde
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val = model.forward(f_val_t, x_pace_tensor, x_tensor_norm)
            val_loss = ((y_val - u_val_t)**2).mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)

        if epoch % 500 == 0:
            print(f"  Fold {fold_id} Epoch {epoch}: train={avg_train_loss:.6f} "
                  f"val={val_loss:.6f} best={best_val_loss:.6f}", flush=True)

    # Load best model and evaluate
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_norm = to_numpy(model(f_test_t, x_pace_tensor, x_tensor_norm))
    pred_phy = pred_norm * u_std_train + u_mean_train

    # Per-case metrics (pooled across 9 pacing sites)
    l2_errors, mae_errors = [], []
    for i in range(len(test_idx)):
        for s in range(pacing.shape[0]):  # 9 pacing sites
            u_p = pred_phy[i, s]
            u_t = u_test_raw[i, s]
            l2 = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
            mae = np.mean(np.abs(u_p - u_t))
            l2_errors.append(l2)
            mae_errors.append(mae)

    return l2_errors, mae_errors


def main():
    set_seed(42)

    args = ParseArgument()
    device = args.device
    epochs = args.epochs

    set_alpha = 0.05
    save_dir = f"CV_{N_FOLDS}fold_pinn_v600_alpha{set_alpha}_{epochs}ep"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_path = os.path.join(DATA_BASE, "DIMON_training_data_healthy_fixed.npz")
    data_path2 = os.path.join(DATA_BASE, "reference_cobiveco.npz")

    dataset = np.load(data_path)
    dataset2 = np.load(data_path2)
    theta_all = dataset['theta'].astype(np.float32)       # (125, 60)
    pacing = dataset['pacing'].astype(np.float32)          # (9, 4)
    u_all = dataset['u_data'].astype(np.float32)           # (125, 9, 50797)
    anisotropy = dataset['ref_anisotropy'].astype(np.float32)  # (50797, 9)
    trunk_input = dataset['cartesian_coords'].astype(np.float32)  # (50797, 3)

    n_total = len(theta_all)
    n_pacing = pacing.shape[0]
    print(f"Data: {n_total} hearts, {n_pacing} pacing sites, "
          f"{trunk_input.shape[0]} nodes")
    print(f"{N_FOLDS}-fold CV, {epochs} epochs per fold, alpha={set_alpha}\n")

    # Shuffle heart indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    fold_size = n_total // N_FOLDS  # 25

    all_l2, all_mae = [], []
    fold_l2_means, fold_mae_means = [], []

    tic_total = time.time()
    for fold in range(N_FOLDS):
        print(f"=== Fold {fold} ===")
        tic = time.time()

        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        remain_idx = np.concatenate([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:]
        ])
        val_idx = remain_idx[-5:]
        train_idx = remain_idx[:-5]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, "
              f"Test: {len(test_idx)}")

        l2_list, mae_list = train_fold(
            fold, train_idx, val_idx, test_idx,
            theta_all, pacing, u_all, trunk_input,
            anisotropy, set_alpha, epochs, device, save_dir)

        all_l2.extend(l2_list)
        all_mae.extend(mae_list)
        fold_l2_means.append(np.mean(l2_list))
        fold_mae_means.append(np.mean(mae_list))

        fold_time = int((time.time() - tic) / 60)
        print(f"  Fold {fold}: Rel L2 = {np.mean(l2_list):.4f} +/- "
              f"{np.std(l2_list):.4f}, MAE = {np.mean(mae_list):.2f} +/- "
              f"{np.std(mae_list):.2f} ms ({fold_time} min)\n")

    total_time = int((time.time() - tic_total) / 60)

    # Summary
    print("=" * 65)
    print(f"PINN 5-fold CV summary (alpha={set_alpha}, {epochs} epochs)")
    print(f"  Per-fold Rel L2: " +
          ", ".join(f"{v:.4f}" for v in fold_l2_means))
    print(f"  Per-fold MAE:    " +
          ", ".join(f"{v:.2f}" for v in fold_mae_means))
    print(f"  Pooled Rel L2: {np.mean(all_l2):.4f} +/- {np.std(all_l2):.4f}")
    print(f"  Pooled MAE:    {np.mean(all_mae):.2f} +/- "
          f"{np.std(all_mae):.2f} ms")
    print(f"  Total cases evaluated: {len(all_l2)} "
          f"({len(all_l2) // n_pacing} hearts x {n_pacing} pacings)")
    print(f"  Total time: {total_time} min")

    # Save results
    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(results_path,
                        l2_errors=np.array(all_l2),
                        mae_errors=np.array(all_mae),
                        fold_l2_means=np.array(fold_l2_means),
                        fold_mae_means=np.array(fold_mae_means),
                        fold_indices=indices,
                        alpha=set_alpha)
    print(f"Saved {results_path}")

    # Loss curve per fold is not saved (would require tracking inside
    # train_fold); use main.py for single-run loss curves.


if __name__ == "__main__":
    main()
