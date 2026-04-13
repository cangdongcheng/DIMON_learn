"""
Cobiveco_with_fiber 5-fold cross-validation (3-branch: geo + pacing + trunk).
Trunk: 4D Cobiveco. Data: DIMON_training_data_healthy.npz (125 hearts × 9 pacings).

Shuffles 125 hearts with fixed seed, splits into 5 folds of 25 each.
Each fold: 95 train (90 fit + 5 val) / 25 test.
Reports per-fold and pooled Rel L2 and MAE, broken down by pacing site.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Cobiveco_with_fiber

    python main2_cv.py --epochs 3000 --device cuda
"""
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from opnn import *
import random

N_FOLDS = 5


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(fold_id, train_idx, val_idx, test_idx,
               theta_all, pacing, trunk_raw, u_all,
               epochs, device, save_dir):
    """Train and evaluate one fold. Returns per-case-pacing (l2, mae) arrays."""

    dim_br_geo = [60, 200, 200, 200, 200]
    dim_br_pace = [4, 200, 200, 200, 200]
    dim_tr = [4, 200, 200, 200, 200]
    batch_size = 10
    learning_rate = 0.0005

    # Split by heart index
    f_train = theta_all[train_idx]
    u_train_raw = u_all[train_idx]    # (n_train, 9, 50797)
    f_val = theta_all[val_idx]
    u_val_raw = u_all[val_idx]
    f_test = theta_all[test_idx]
    u_test_raw = u_all[test_idx]

    # Trunk (shared across all hearts)
    x_tensor_raw = torch.tensor(trunk_raw, dtype=torch.float).to(device)
    x_pace_tensor = torch.tensor(pacing, dtype=torch.float).to(device)

    # Normalization (per-fold, training stats only)
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

    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    # Training
    for epoch in range(epochs):
        model.train()
        for b_f, b_u in train_loader:
            optimizer.zero_grad()
            loss = ((model(b_f, x_pace_tensor, x_tensor) - b_u) ** 2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = ((model(f_val_t, x_pace_tensor, x_tensor) - u_val_t) ** 2).mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)

        if epoch % 500 == 0:
            print(f"  Fold {fold_id} Epoch {epoch}: val_loss={val_loss:.6f} "
                  f"best={best_val_loss:.6f}", flush=True)

    # Load best and evaluate
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_norm = to_numpy(model(f_test_t, x_pace_tensor, x_tensor))
    pred_phy = pred_norm * u_std_train + u_mean_train  # (n_test, 9, 50797)

    # Per heart × pacing metrics
    n_test = len(test_idx)
    l2_errors = np.zeros((n_test, 9))
    mae_errors = np.zeros((n_test, 9))
    for i in range(n_test):
        for s in range(9):
            u_p = pred_phy[i, s]
            u_t = u_test_raw[i, s]
            l2_errors[i, s] = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
            mae_errors[i, s] = np.mean(np.abs(u_p - u_t))

    return l2_errors, mae_errors


def main():
    set_seed(42)

    args = ParseArgument()
    device = args.device
    epochs = args.epochs

    save_dir = f"CV_{N_FOLDS}fold_{epochs}ep"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_path = "../DIMON_training_data_healthy.npz"
    data_path2 = "../reference_cobiveco.npz"
    dataset = np.load(data_path)
    dataset2 = np.load(data_path2)

    theta_all = dataset['theta'].astype(np.float32)        # (125, 60)
    pacing = dataset['pacing'].astype(np.float32)           # (9, 4)
    u_all = dataset['u_data'].astype(np.float32)            # (125, 9, 50797)
    cobiveco_coords = dataset2['cobiveco'].astype(np.float32)
    trunk_raw = cobiveco_coords[:, :4]                      # (50797, 4)
    cartesian_coords = dataset['cartesian_coords']          # for plotting

    n_total = theta_all.shape[0]
    print(f"Data: {n_total} hearts, 9 pacings, {trunk_raw.shape[0]} nodes")
    print(f"{N_FOLDS}-fold CV, {epochs} epochs per fold\n")

    # Shuffle heart indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    fold_size = n_total // N_FOLDS  # 25

    all_l2 = []   # list of (n_test, 9) arrays
    all_mae = []
    all_test_idx = []

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

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        l2, mae = train_fold(fold, train_idx, val_idx, test_idx,
                             theta_all, pacing, trunk_raw, u_all,
                             epochs, device, save_dir)

        all_l2.append(l2)
        all_mae.append(mae)
        all_test_idx.append(test_idx)

        fold_time = int((time.time() - tic) / 60)
        # Pool across pacings for fold summary
        print(f"  Fold {fold}: Rel L2 = {l2.mean():.4f} +/- {l2.std():.4f}, "
              f"MAE = {mae.mean():.2f} +/- {mae.std():.2f} ms "
              f"({fold_time} min)\n")

    total_time = int((time.time() - tic_total) / 60)

    # Combine all folds
    all_l2 = np.concatenate(all_l2, axis=0)     # (125, 9)
    all_mae = np.concatenate(all_mae, axis=0)     # (125, 9)
    all_test_idx = np.concatenate(all_test_idx)   # (125,)

    # Per-pacing stats
    print("=" * 65)
    print(f"{'Pacing':<10} {'Rel L2':>15} {'MAE (ms)':>15}")
    print("-" * 42)
    for s in range(9):
        print(f"  {s:<8} {all_l2[:, s].mean():.4f} +/- {all_l2[:, s].std():.4f}"
              f"    {all_mae[:, s].mean():.2f} +/- {all_mae[:, s].std():.2f}")
    print("-" * 42)
    print(f"  {'Pooled':<8} {all_l2.mean():.4f} +/- {all_l2.std():.4f}"
          f"    {all_mae.mean():.2f} +/- {all_mae.std():.2f}")
    print(f"\nTotal time: {total_time} min")

    # Save
    # Map back to case order for per-heart summary
    case_names_ordered = []
    try:
        import csv
        DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"
        cases_canonical = []
        with open(os.path.join(DATA_BASE, "valid_list.csv")) as f:
            for row in csv.DictReader(f):
                if row["to_canonical"] == "1":
                    cases_canonical.append(row["filename"])
        case_names_ordered = [cases_canonical[i] for i in all_test_idx]
    except Exception:
        case_names_ordered = [str(i) for i in all_test_idx]

    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(results_path,
                        case_names=np.array(case_names_ordered),
                        l2_errors=all_l2,
                        mae_errors=all_mae,
                        fold_indices=indices)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
