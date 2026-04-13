"""
Geo-DeepONet 5-fold cross-validation.

Shuffles the 125 cases with a fixed seed, splits into 5 folds of 25 each.
Each fold: 100 train (80 fit + 20 val for early stopping) / 25 test.
Reports per-fold and pooled Rel L2 and MAE.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DeepONet

    python main_cv.py --epochs 5000 --device cuda
    python main_cv.py --epochs 50000 --device cuda
"""
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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
               theta_all, x_data, u_all, case_names,
               epochs, device, save_dir):
    """Train and evaluate one fold. Returns per-case (l2, mae) for test set."""

    num_geomode = 60
    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_tr = [4, 200, 200, 200, 200]
    batch_size = 10
    learning_rate = 0.0005

    # Split
    f_train = theta_all[train_idx]
    u_train_raw = u_all[train_idx]
    f_val = theta_all[val_idx]
    u_val_raw = u_all[val_idx]
    f_test = theta_all[test_idx]
    u_test_raw = u_all[test_idx]

    x_tensor_raw = torch.tensor(x_data, dtype=torch.float).to(device)

    # Normalization (per-fold, from training set only)
    x_min = x_tensor_raw.min(dim=0, keepdim=True)[0]
    x_max = x_tensor_raw.max(dim=0, keepdim=True)[0]
    x_tensor = (x_tensor_raw - x_min) / (x_max - x_min + 1e-8)

    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_norm = (f_train - f_mean) / f_std
    f_val_norm = (f_val - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

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

    train_ds = TensorDataset(f_train_t, u_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = opnn(dim_br_geo, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        for b_f, b_u in train_loader:
            optimizer.zero_grad()
            loss = ((model(b_f, x_tensor) - b_u) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = ((model(f_val_t, x_tensor) - u_val_t) ** 2).mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)

        if epoch % 500 == 0:
            print(f"  Fold {fold_id} Epoch {epoch}: val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f}", flush=True)

    # Load best model and evaluate
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_norm = to_numpy(model(f_test_t, x_tensor))
    pred_phy = pred_norm * u_std_train + u_mean_train

    # Per-case metrics
    l2_errors, mae_errors = [], []
    for i in range(len(test_idx)):
        u_p = pred_phy[i]
        u_t = u_test_raw[i]
        l2 = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
        mae = np.mean(np.abs(u_p - u_t))
        l2_errors.append(l2)
        mae_errors.append(mae)

    return l2_errors, mae_errors, pred_phy, u_test_raw


def main():
    set_seed(42)

    args = ParseArgument()
    device = args.device
    epochs = args.epochs

    save_dir = f"CV_{N_FOLDS}fold_{epochs}ep"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    dataset = np.load(os.path.join(DATA_BASE, "geo_deeponet_data.npz"),
                      allow_pickle=True)
    theta_all = dataset['theta'].astype(np.float32)
    x_data = dataset['coords'].astype(np.float32)
    u_all = dataset['at'].astype(np.float32)
    case_names = dataset['case_names']

    n_total = len(theta_all)
    print(f"Data: {n_total} cases, {x_data.shape[0]} nodes")
    print(f"{N_FOLDS}-fold CV, {epochs} epochs per fold\n")

    # Shuffle indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    fold_size = n_total // N_FOLDS  # 25

    all_l2, all_mae = [], []
    all_case_results = []

    tic_total = time.time()
    for fold in range(N_FOLDS):
        print(f"=== Fold {fold} ===")
        tic = time.time()

        # Test: this fold's 25 cases
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        # Train + val: remaining 100
        remain_idx = np.concatenate([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:]
        ])
        # Use last 5 of remaining as validation
        val_idx = remain_idx[-5:]
        train_idx = remain_idx[:-5]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        l2_list, mae_list, pred, true = train_fold(
            fold, train_idx, val_idx, test_idx,
            theta_all, x_data, u_all, case_names,
            epochs, device, save_dir)

        all_l2.extend(l2_list)
        all_mae.extend(mae_list)

        # Print fold results
        for i, ti in enumerate(test_idx):
            all_case_results.append((case_names[ti], l2_list[i], mae_list[i]))

        fold_time = int((time.time() - tic) / 60)
        print(f"  Fold {fold}: Rel L2 = {np.mean(l2_list):.4f} +/- {np.std(l2_list):.4f}, "
              f"MAE = {np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f} ms "
              f"({fold_time} min)\n")

    total_time = int((time.time() - tic_total) / 60)

    # Summary
    print("=" * 65)
    print(f"{'Case':<35} {'Rel L2':>10} {'MAE (ms)':>10}")
    print("-" * 58)
    for name, l2, mae in sorted(all_case_results):
        print(f"{name:<35} {l2:10.4f} {mae:10.2f}")
    print("-" * 58)
    print(f"{'Pooled Mean':<35} {np.mean(all_l2):10.4f} {np.mean(all_mae):10.2f}")
    print(f"{'Pooled Std':<35} {np.std(all_l2):10.4f} {np.std(all_mae):10.2f}")
    print(f"\nTotal time: {total_time} min")

    # Save results
    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(results_path,
                        case_names=np.array([r[0] for r in all_case_results]),
                        l2_errors=np.array([r[1] for r in all_case_results]),
                        mae_errors=np.array([r[2] for r in all_case_results]),
                        fold_indices=indices)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
