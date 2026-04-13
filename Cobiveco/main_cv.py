"""
Flattened Cobiveco 5-fold cross-validation.

Flattens (125 hearts x 9 pacings) into 1125 independent samples.
Each fold: 25 test hearts (225 samples), 5 val hearts (45 samples),
95 train hearts (855 samples). Shuffled by heart, not by sample —
all 9 pacings for a heart stay in the same fold.

This is the flattened-data baseline to compare against the stacked
3-branch MIONet in Cobiveco_with_fiber/.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Cobiveco

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
               theta_all, pacing_all, u_all, cobiveco_coords,
               epochs, device, save_dir):
    """Train and evaluate one fold. Returns per-(heart,pacing) L2 and MAE."""

    dim_br_geo  = [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200]
    dim_tr      = [4, 200, 200, 200, 200]
    batch_size = 95
    learning_rate = 0.0005
    num_sims = pacing_all.shape[0]  # 9
    num_pts = cobiveco_coords.shape[0]

    # Flatten: each (heart, pacing) becomes an independent sample
    def flatten_split(heart_idx):
        theta_flat = np.repeat(theta_all[heart_idx], num_sims, axis=0)
        pacing_flat = np.tile(pacing_all, (len(heart_idx), 1))
        u_flat = u_all[heart_idx].reshape(-1, num_pts)
        return theta_flat, pacing_flat, u_flat

    f_train, pace_train, u_train_raw = flatten_split(train_idx)
    f_val, pace_val, u_val_raw = flatten_split(val_idx)
    f_test, pace_test, u_test_raw = flatten_split(test_idx)

    # Trunk: min-max normalization
    x_tensor_raw = torch.tensor(cobiveco_coords, dtype=torch.float).to(device)
    x_min = x_tensor_raw.min(dim=0, keepdim=True)[0]
    x_max = x_tensor_raw.max(dim=0, keepdim=True)[0]
    x_tensor = (x_tensor_raw - x_min) / (x_max - x_min + 1e-8)

    # Geo branch: z-score (per-fold, from training set only)
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_norm = (f_train - f_mean) / f_std
    f_val_norm = (f_val - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

    # Labels: min-shift + std scaling
    u_mean = u_train_raw.min()
    u_std = u_train_raw.std()
    u_train_norm = (u_train_raw - u_mean) / u_std
    u_val_norm = (u_val_raw - u_mean) / u_std

    # Tensors
    f_train_t = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    u_train_t = torch.tensor(u_train_norm, dtype=torch.float).to(device)
    pace_train_t = torch.tensor(pace_train, dtype=torch.float).to(device)
    f_val_t = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    u_val_t = torch.tensor(u_val_norm, dtype=torch.float).to(device)
    pace_val_t = torch.tensor(pace_val, dtype=torch.float).to(device)
    f_test_t = torch.tensor(f_test_norm, dtype=torch.float).to(device)
    pace_test_t = torch.tensor(pace_test, dtype=torch.float).to(device)

    train_ds = TensorDataset(f_train_t, pace_train_t, u_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for b_f, b_pace, b_u in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(b_f, b_pace, x_tensor)
            loss = ((y_pred - b_u)**2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val = model.forward(f_val_t, pace_val_t, x_tensor)
            val_loss = ((y_val - u_val_t)**2).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)

        if epoch % 500 == 0:
            avg_train = epoch_loss / len(train_loader)
            print(f"  Fold {fold_id} Epoch {epoch}: train={avg_train:.6f} "
                  f"val={val_loss:.6f} best={best_val_loss:.6f}", flush=True)

    # Load best model and evaluate
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_norm = to_numpy(model.forward(f_test_t, pace_test_t, x_tensor))
    pred_phy = pred_norm * u_std + u_mean

    # Per-(heart, pacing) metrics
    l2_errors, mae_errors = [], []
    n_test = len(test_idx)
    pred_3d = pred_phy.reshape(n_test, num_sims, num_pts)
    true_3d = u_test_raw.reshape(n_test, num_sims, num_pts)

    for i in range(n_test):
        for s in range(num_sims):
            u_p = pred_3d[i, s]
            u_t = true_3d[i, s]
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

    save_dir = f"CV_{N_FOLDS}fold_flat_cobiveco4d_{epochs}ep"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    dataset = np.load(os.path.join(DATA_BASE,
                      "DIMON_training_data_healthy_fixed.npz"))
    theta_all = dataset['theta'].astype(np.float32)       # (125, 60)
    pacing_all = dataset['pacing'].astype(np.float32)     # (9, 4)
    u_all = dataset['u_data'].astype(np.float32)          # (125, 9, 50797)
    cobiveco_coords = dataset['cobiveco'].astype(np.float32)  # (50797, 4)

    n_total = len(theta_all)
    n_pacing = pacing_all.shape[0]
    print(f"Data: {n_total} hearts, {n_pacing} pacing sites, "
          f"{cobiveco_coords.shape[0]} nodes")
    print(f"Flattened: {n_total * n_pacing} independent samples")
    print(f"{N_FOLDS}-fold CV, {epochs} epochs per fold\n")

    # Shuffle heart indices (not sample indices — keep pacings grouped)
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

        print(f"  Train: {len(train_idx)} hearts ({len(train_idx)*n_pacing} samples), "
              f"Val: {len(val_idx)} hearts, Test: {len(test_idx)} hearts")

        l2_list, mae_list = train_fold(
            fold, train_idx, val_idx, test_idx,
            theta_all, pacing_all, u_all, cobiveco_coords,
            epochs, device, save_dir)

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
    print(f"Flattened Cobiveco 5-fold CV summary ({epochs} epochs)")
    print(f"  Per-fold Rel L2: " +
          ", ".join(f"{v:.4f}" for v in fold_l2_means))
    print(f"  Per-fold MAE:    " +
          ", ".join(f"{v:.2f}" for v in fold_mae_means))
    print(f"  Pooled Rel L2: {np.mean(all_l2):.4f} +/- {np.std(all_l2):.4f}")
    print(f"  Pooled MAE:    {np.mean(all_mae):.2f} +/- "
          f"{np.std(all_mae):.2f} ms")
    print(f"  Total evaluations: {len(all_l2)} "
          f"({len(all_l2) // n_pacing} hearts x {n_pacing} pacings)")
    print(f"  Total time: {total_time} min")

    # Save results
    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(results_path,
                        l2_errors=np.array(all_l2),
                        mae_errors=np.array(all_mae),
                        fold_l2_means=np.array(fold_l2_means),
                        fold_mae_means=np.array(fold_mae_means),
                        fold_indices=indices)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()