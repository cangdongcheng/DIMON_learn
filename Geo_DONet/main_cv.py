"""
Geo-DONet 5-fold cross-validation (geometry → V_m(x,t)).

Trunk: 4D Cobiveco + 1D normalised time (or 3D xyz + 1D time, via --trunk xyz).
Data: $DATA_BASE/geo_donet_data_f121.npz (125 hearts, 50797 nodes, ~121 frames).

Fold layout (matches Cobiveco_with_fiber/main_cv.py):
  shuffle 125 hearts with seed=42, split into 5 folds of 25.
  Per fold: 100 train (95 fit + 5 val) / 25 test.
  Per-fold normalisation uses training stats only.
  Best-val checkpoint per fold; metrics computed on the held-out 25.

Reports per-fold and pooled:
  V_m  Rel L2 + MAE (mV)
  AT   Rel L2 + MAE (ms)        — AT = first-time V_m crosses -10 mV from below

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DONet

    python main_cv.py --epochs 5000 --width 300 --batch-size 24 \\
                      --device cuda --lr-schedule
"""
import os
import time as timer
import torch
import numpy as np
import random

from utils import ParseArgument, to_numpy
from opnn import opnn

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"
N_FOLDS = 5
NUM_VAL = 5
AT_THRESHOLD_MV = -10.0


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precompute_trunk_chunks(coords, time_points, chunk_size, device):
    """Same as main.py: build (M*T_c, D+1) trunk grids per temporal chunk."""
    M, D = coords.shape
    T = len(time_points)
    chunks = []
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        T_c = t_end - t_start
        coords_rep = coords.unsqueeze(1).expand(M, T_c, D).reshape(M * T_c, D)
        time_rep = time_points[t_start:t_end].unsqueeze(0).expand(M, T_c).reshape(M * T_c, 1)
        chunks.append(torch.cat([coords_rep, time_rep], dim=1))
    return chunks


def chunked_forward(model, f_geo, trunk_chunks, M):
    """Same as main.py: branch once, evaluate trunk per chunk, concat."""
    N = f_geo.shape[0]
    y_br = model._branch_g(f_geo)
    outputs = []
    for xt_chunk in trunk_chunks:
        T_c = xt_chunk.shape[0] // M
        y_tr = model.encode_trunk(xt_chunk)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr)
        outputs.append(y_out.reshape(N, M, T_c))
    return torch.cat(outputs, dim=2)


def compute_at(vm, time_ms, threshold=AT_THRESHOLD_MV):
    """First crossing of `threshold` per node.

    vm:      (N, M, T) physical units (mV)
    time_ms: (T,) physical times in ms
    Returns AT (N, M) in ms; -1 where the node never crosses.
    """
    crossed = vm > threshold                              # (N, M, T)
    first = crossed.argmax(axis=-1)                       # (N, M); 0 if never True
    never = ~crossed.any(axis=-1)                         # mask of never-activated
    at = time_ms[first].astype(np.float32)
    at[never] = -1.0
    return at


def train_fold(fold_id, train_idx, val_idx, test_idx,
               theta_all, vm_all, coords_np, time_norm_np, time_ms,
               args, save_dir):
    """Train one fold, evaluate on its 25 test cases. Returns metrics dict."""
    device = args.device
    epochs = args.epochs
    width = args.width
    batch_size = args.batch_size

    num_geomode = 60
    dim_br_geo = [num_geomode] + [width] * 4
    trunk_spatial_dim = coords_np.shape[1]
    dim_tr = [trunk_spatial_dim + 1] + [width] * 4

    learning_rate = 0.001 if args.lr_schedule else 0.0005
    time_chunk_size = 25

    n_pts = coords_np.shape[0]
    n_time = len(time_ms)

    # --- split ---
    f_train = theta_all[train_idx]
    vm_train_raw = vm_all[train_idx]
    f_val = theta_all[val_idx]
    vm_val_raw = vm_all[val_idx]
    f_test = theta_all[test_idx]
    vm_test_raw = vm_all[test_idx]

    # --- normalisation (training stats only) ---
    coords_t = torch.tensor(coords_np, dtype=torch.float).to(device)
    x_min = coords_t.min(dim=0, keepdim=True)[0]
    x_max = coords_t.max(dim=0, keepdim=True)[0]
    coords_norm = (coords_t - x_min) / (x_max - x_min + 1e-8)
    time_t = torch.tensor(time_norm_np, dtype=torch.float).to(device)

    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_n = (f_train - f_mean) / f_std
    f_val_n = (f_val - f_mean) / f_std
    f_test_n = (f_test - f_mean) / f_std

    vm_mean = vm_train_raw.min()
    vm_std = vm_train_raw.std()
    vm_train_n = (vm_train_raw - vm_mean) / vm_std
    vm_val_n = (vm_val_raw - vm_mean) / vm_std

    # --- tensors ---
    f_train_t = torch.tensor(f_train_n, dtype=torch.float).to(device)
    f_val_t = torch.tensor(f_val_n, dtype=torch.float).to(device)
    f_test_t = torch.tensor(f_test_n, dtype=torch.float).to(device)
    vm_train_t = torch.tensor(vm_train_n, dtype=torch.float).to(device)
    vm_val_t = torch.tensor(vm_val_n, dtype=torch.float).to(device)

    trunk_chunks = precompute_trunk_chunks(coords_norm, time_t,
                                           time_chunk_size, device)

    # --- model ---
    model = opnn(dim_br_geo, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if args.lr_schedule:
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                             total_iters=epochs)

    model_path = os.path.join(save_dir, f"model_fold{fold_id}.pt")
    best_val_loss = float('inf')

    n_train = len(train_idx)
    train_indices = np.arange(n_train)

    # --- train ---
    tic = timer.time()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_indices)
        for b_start in range(0, n_train, batch_size):
            b_idx = train_indices[b_start:b_start + batch_size]
            b_f = f_train_t[b_idx]
            b_vm = vm_train_t[b_idx]
            optimizer.zero_grad()
            pred = chunked_forward(model, b_f, trunk_chunks, n_pts)
            loss = ((pred - b_vm) ** 2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = chunked_forward(model, f_val_t, trunk_chunks, n_pts)
            val_loss = ((pred_val - vm_val_t) ** 2).mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_path)
        if scheduler is not None:
            scheduler.step()

        if epoch % 200 == 0:
            elapsed = timer.time() - tic
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
            print(f"  Fold {fold_id} Epoch {epoch}/{epochs}: "
                  f"val={val_loss:.6f} best={best_val_loss:.6f} "
                  f"ETA {int(eta // 60)}m{int(eta % 60)}s",
                  flush=True)

    fold_min = int((timer.time() - tic) / 60)

    # --- evaluate on test set ---
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_test = len(test_idx)
    pred_phy = np.zeros((n_test, n_pts, n_time), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_test):
            pred_n = chunked_forward(model, f_test_t[i:i + 1], trunk_chunks, n_pts)
            pred_phy[i] = to_numpy(pred_n[0]) * vm_std + vm_mean

    # V_m metrics per case
    vm_l2 = np.zeros(n_test)
    vm_mae = np.zeros(n_test)
    for i in range(n_test):
        p, t = pred_phy[i], vm_test_raw[i]
        vm_l2[i] = np.linalg.norm(p - t) / np.linalg.norm(t)
        vm_mae[i] = np.mean(np.abs(p - t))

    # AT metrics per case (intersection of activated nodes only)
    at_pred = compute_at(pred_phy, time_ms)
    at_true = compute_at(vm_test_raw, time_ms)
    at_l2 = np.full(n_test, np.nan, dtype=np.float32)
    at_mae = np.full(n_test, np.nan, dtype=np.float32)
    for i in range(n_test):
        mask = (at_pred[i] >= 0) & (at_true[i] >= 0)
        if mask.sum() == 0:
            continue
        p, t = at_pred[i][mask], at_true[i][mask]
        at_l2[i] = np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12)
        at_mae[i] = np.mean(np.abs(p - t))

    # Free large GPU tensors before returning
    del f_train_t, f_val_t, f_test_t, vm_train_t, vm_val_t, trunk_chunks
    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'vm_l2': vm_l2, 'vm_mae': vm_mae,
        'at_l2': at_l2, 'at_mae': at_mae,
        'fold_time_min': fold_min,
        'best_val_loss': best_val_loss,
    }


def main():
    set_seed(42)
    args = ParseArgument()
    epochs = args.epochs

    lr_tag = '_lrsched' if args.lr_schedule else ''
    trunk_tag = '' if args.trunk == 'cobiveco' else f'_{args.trunk}'
    save_dir = f"CV_{N_FOLDS}fold_{epochs}ep_w{args.width}{lr_tag}{trunk_tag}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    # --- load data ---
    data_path = os.path.join(DATA_BASE, "geo_donet_data_f121.npz")
    data = np.load(data_path, allow_pickle=True)
    theta = data['theta'][:, :60].astype(np.float32)              # (125, 60)
    cobiveco_coords = data['coords'].astype(np.float32)            # (50797, 4)
    vm_all = data['vm'].astype(np.float32)                         # (125, 50797, T)
    time_ms = data['time'].astype(np.float32)                      # (T,)
    case_names = data['case_names']

    if args.trunk == 'xyz':
        dimon_path = os.path.join(os.path.dirname(__file__), "..",
                                  "DIMON_training_data_healthy.npz")
        coords_np = np.load(dimon_path)['cartesian_coords'].astype(np.float32)
        print(f"Trunk: 3D Cartesian xyz from {dimon_path}")
    else:
        coords_np = cobiveco_coords
        print("Trunk: 4D Cobiveco (ab, rt, tm, tv)")

    n_total = theta.shape[0]
    t_min, t_max = time_ms.min(), time_ms.max()
    time_norm_np = (time_ms - t_min) / (t_max - t_min + 1e-8)

    print(f"Data: {n_total} hearts, {coords_np.shape[0]} nodes, "
          f"{len(time_ms)} frames, dt={time_ms[1]-time_ms[0]:.0f} ms")
    print(f"{N_FOLDS}-fold CV, {epochs} epochs/fold, width={args.width}, "
          f"lr_schedule={args.lr_schedule}\n")

    # --- shuffle hearts (seed=42 from set_seed) ---
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    fold_size = n_total // N_FOLDS

    folds_info = []
    tic_total = timer.time()
    for fold in range(N_FOLDS):
        print(f"=== Fold {fold} ===")
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        remain_idx = np.concatenate([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:],
        ])
        val_idx = remain_idx[-NUM_VAL:]
        train_idx = remain_idx[:-NUM_VAL]
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, "
              f"Test: {len(test_idx)}")

        result = train_fold(fold, train_idx, val_idx, test_idx,
                            theta, vm_all, coords_np, time_norm_np, time_ms,
                            args, save_dir)
        result['test_idx'] = test_idx
        folds_info.append(result)

        print(f"  Fold {fold}: "
              f"V_m L2={result['vm_l2'].mean():.4f} ± {result['vm_l2'].std():.4f}, "
              f"MAE={result['vm_mae'].mean():.2f} ± {result['vm_mae'].std():.2f} mV  "
              f"AT L2={np.nanmean(result['at_l2']):.4f} ± {np.nanstd(result['at_l2']):.4f}, "
              f"MAE={np.nanmean(result['at_mae']):.2f} ± {np.nanstd(result['at_mae']):.2f} ms  "
              f"({result['fold_time_min']} min)\n")

    total_min = int((timer.time() - tic_total) / 60)

    # --- pool ---
    all_vm_l2 = np.concatenate([f['vm_l2'] for f in folds_info])
    all_vm_mae = np.concatenate([f['vm_mae'] for f in folds_info])
    all_at_l2 = np.concatenate([f['at_l2'] for f in folds_info])
    all_at_mae = np.concatenate([f['at_mae'] for f in folds_info])
    all_test_idx = np.concatenate([f['test_idx'] for f in folds_info])

    # --- summary ---
    print("=" * 65)
    print(f"{N_FOLDS}-fold pooled (N={len(all_vm_l2)} cases):")
    print(f"  V_m  Rel L2 = {all_vm_l2.mean():.4f} ± {all_vm_l2.std():.4f}")
    print(f"  V_m  MAE    = {all_vm_mae.mean():.2f} ± {all_vm_mae.std():.2f} mV")
    print(f"  AT   Rel L2 = {np.nanmean(all_at_l2):.4f} ± {np.nanstd(all_at_l2):.4f}")
    print(f"  AT   MAE    = {np.nanmean(all_at_mae):.2f} ± {np.nanstd(all_at_mae):.2f} ms")
    print()
    print("Per-fold:")
    for k, f in enumerate(folds_info):
        print(f"  Fold {k}: V_m L2={f['vm_l2'].mean():.4f} ± {f['vm_l2'].std():.4f}, "
              f"AT L2={np.nanmean(f['at_l2']):.4f} ± {np.nanstd(f['at_l2']):.4f}, "
              f"{f['fold_time_min']} min")
    print(f"\nTotal time: {total_min} min")

    # --- save ---
    case_names_ordered = np.array(
        [str(case_names[i]) for i in all_test_idx], dtype=object,
    )
    results_path = os.path.join(save_dir, "cv_results.npz")
    np.savez_compressed(
        results_path,
        case_names=case_names_ordered,
        test_idx=all_test_idx,
        fold_indices=indices,
        vm_l2=all_vm_l2,
        vm_mae=all_vm_mae,
        at_l2=all_at_l2,
        at_mae=all_at_mae,
    )
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
