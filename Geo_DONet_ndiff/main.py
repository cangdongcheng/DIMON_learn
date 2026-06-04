"""
Geo-DONet-ndiff: geometry → V_m(x,t) on the canonical reference, trained with
MSE + a signed-Laplacian neighbour-difference loss. The extra term matches
Vm[i] - mean(neighbour Vm) between prediction and GT on the verified
canonical-order mesh, i.e. it penalises wrong local spatial structure that
pointwise MSE cannot see. --ndiff-weight 0 reproduces the Geo_DONet baseline.

DeepONet with 1 branch + 1 trunk:
  - Branch (geometry): theta (N, 60) PCA coefficients → (N, p)
  - Trunk (spatiotemporal): (ab, rt, tm, tv, t_norm) — plain 5D, no Fourier
    features. Trunk input dim = 5 (4 Cobiveco + normalised time).
  - Output: einsum(branch, trunk) → V_m(x,t) on reference mesh (N, M, T).

Temporal chunking: the full M×T grid (~6M points at 121 frames) is split
into chunks of T_c frames. The trunk is evaluated per chunk, and the
branch output is reused across chunks.

Data: geo_donet_data_f121.npz from package_training_data.py

Evaluation outputs (--test-model 1) — all SVG, transparent background:
  Test/
    vm_traces_heart{h}_{GT,Pred,combined}.svg   — 5-node V_m time-series
    snapshots/heart{h}_t{XXX}ms_{GT,Pred,AbsErr}.svg
                                                — 3D V_m at frames selected by --vm-frames
    activation_time/heart{h}_AT_{GT,Pred,AbsErr}.svg
                                                — AT map (first -10 mV crossing)
    colorbars/cbar_{Vm,Vm_AbsErr,AT,AT_AbsErr}.svg
                                                — standalone shared colorbars
    test_predictions.npz                        — predictions + AT metrics

Metrics reported: V_m Rel L2 + MAE; AT Rel L2 + MAE (per case, mean ± std).
3D scatters are rasterised inside the SVG (full 50 797 nodes @ 300 dpi).
Rendering is parallelised across a ProcessPoolExecutor (up to 8 workers).

Eval-time CLI flags:
    --skip-snapshots           skip 3D V_m + AT scatters (keeps traces + metrics)
    --vm-frames START:END:STEP V_m snapshot time range, ms (default 0:300:10).
                               END inclusive. e.g. 0:600:10 for the full beat.

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/Geo_DONet

    # Train
    python main.py --epochs 5000 --device cuda

    # Evaluate (single line, no backslashes when pasting into bash)
    python main.py --test-model 1 --device cuda --epochs 5000 --width 300 --model-path CheckPts/model_chkpts_geo_donet_5000ep_0.0005lr_w300.pt

    # Evaluate, full-beat snapshots every 10 ms
    python main.py --test-model 1 --device cuda --epochs 5000 --width 300 --model-path CheckPts/model_chkpts_geo_donet_5000ep_0.0005lr_w300.pt --vm-frames 0:600:10

    # Quick eval (metrics + traces only, no 3D scatters)
    python main.py --test-model 1 --device cuda --epochs 5000 --width 300 --model-path CheckPts/model_chkpts_geo_donet_5000ep_0.0005lr_w300.pt --skip-snapshots
"""
import os
import torch
import time as timer
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from opnn import *
import matplotlib.pyplot as plt
import random
from concurrent.futures import ProcessPoolExecutor


# ── Module-level worker for parallel 3D scatter rendering ──────────────────
# ProcessPoolExecutor requires a top-level (picklable) function. We pass the
# shared cartesian_coords via the pool initializer to avoid re-serializing it
# for every task.
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

DATA_BASE = os.environ.get(
    "DIMON_DATA_BASE",
    "/home/svu/e1032484/scratch",  # Vanda default
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precompute_trunk_chunks(coords, time_points, chunk_size, device):
    """
    Precompute (M*T_c, D+1) trunk input grids for all temporal chunks, where
    D = coords.shape[1] (4 for Cobiveco, 3 for xyz).
    Returns: list of (M*T_c, D+1) tensors on device.
    """
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
    """
    Forward pass with precomputed temporal chunks.
    f_geo:        (N, geo_modes) — geometry branch input
    trunk_chunks: list of (M*T_c, 5) precomputed trunk grids on device
    M:            number of spatial nodes
    Returns: (N, M, T)
    """
    N = f_geo.shape[0]

    # Geometry branch — compute once
    y_br = model._branch_g(f_geo)  # (N, p)

    outputs = []
    for xt_chunk in trunk_chunks:
        T_c = xt_chunk.shape[0] // M
        y_tr = model.encode_trunk(xt_chunk)  # (M*T_c, p)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr)  # (N, M*T_c)
        outputs.append(y_out.reshape(N, M, T_c))

    return torch.cat(outputs, dim=2)  # (N, M, T)


def build_laplacian_operator(adj_path, n_nodes, device):
    """Row-normalised adjacency Ahat = D^-1 A (sparse CSR) on the canonical-order
    mesh, so (Ahat @ vm)[i] = mean of vm over node i's neighbours. The signed
    neighbour difference is then L vm = vm - Ahat vm (a graph Laplacian)."""
    z = np.load(adj_path)
    assert int(z['n_nodes']) == n_nodes, \
        f"adjacency n_nodes={int(z['n_nodes'])} != data n_pts={n_nodes}"
    src = z['edge_src'].astype(np.int64)
    dst = z['edge_dst'].astype(np.int64)
    rows = np.concatenate([src, dst])            # both directions (undirected)
    cols = np.concatenate([dst, src])
    deg = np.bincount(rows, minlength=n_nodes).astype(np.float32)
    vals = (1.0 / deg[rows]).astype(np.float32)
    idx = torch.from_numpy(np.stack([rows, cols])).long()
    Ahat = torch.sparse_coo_tensor(idx, torch.from_numpy(vals),
                                   (n_nodes, n_nodes)).coalesce()
    print(f"Laplacian operator: {n_nodes} nodes, {src.shape[0]} edges, "
          f"deg[min={int(deg.min())},max={int(deg.max())}]")
    return Ahat.to_sparse_csr().to(device)


def signed_laplacian_loss(err, Ahat):
    """Mean over (B, M, T) of (L err)^2 with L = I - Ahat, err = pred - target.
    Equivalent to matching signed neighbour diff (Vm - mean neighbours) pred vs GT."""
    B, M, T = err.shape
    e = err.permute(1, 0, 2).reshape(M, B * T)       # (M, B*T)
    le = e - torch.sparse.mm(Ahat, e)                # (M, B*T)
    return (le ** 2).mean()


def main():
    args = ParseArgument()
    set_seed(args.seed)

    device = args.device
    epochs = args.epochs
    test_model = args.test_model
    width = args.width
    batch_size = args.batch_size

    # --- Configuration ---
    num_geomode = 60
    dim_br_geo = [num_geomode] + [width] * 4  # branch: theta → (N, width)

    # Trunk dimension: 4 (Cobiveco) or 3 (xyz), plus 1 for time
    trunk_spatial_dim = 3 if args.trunk == 'xyz' else 4
    dim_tr = [trunk_spatial_dim + 1] + [width] * 4   # trunk: (coords, t)

    learning_rate = 0.001 if args.lr_schedule else 0.0005
    frame_step = 1  # data already subsampled at packaging time
    time_chunk_size = 25  # frames per chunk during forward pass

    lr_tag = '_lrsched' if args.lr_schedule else ''
    trunk_tag = '' if args.trunk == 'cobiveco' else f'_{args.trunk}'
    # Seed tag is omitted for seed=42 to keep the historical run path stable;
    # any other seed gets an explicit tag so checkpoints don't collide.
    seed_tag = '' if args.seed == 42 else f'_seed{args.seed}'
    ndiff_tag = f'_ndiff{args.ndiff_weight:g}' if args.ndiff_weight > 0 else ''
    save_directory = f'geo_donet_{epochs}ep_w{width}{lr_tag}{trunk_tag}{ndiff_tag}{seed_tag}'

    dump_test = f'./Predictions/{save_directory}/Test/'
    dump_train = f'./Predictions/{save_directory}/Train/'
    model_path = args.model_path or f'CheckPts/model_chkpts_{save_directory}.pt'
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## Load Dataset
    data_path = os.path.join(DATA_BASE, "geo_donet_data_f121.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        print("Run: python package_training_data.py --frame-step 5")
        return

    dataset = np.load(data_path, allow_pickle=True)
    theta = dataset['theta'][:, :num_geomode].astype(np.float32)  # (125, 60)
    cobiveco_coords = dataset['coords'].astype(np.float32)         # (50797, 4)
    vm_all = dataset['vm'].astype(np.float32)                      # (125, 50797, T_full)
    time_ms = dataset['time'].astype(np.float32)                   # (T_full,)
    case_names = dataset['case_names']

    # Trunk coordinate choice: 4D Cobiveco or 3D Cartesian xyz of the canonical mesh
    if args.trunk == 'xyz':
        dimon_path = os.path.join(os.path.dirname(__file__), "..",
                                  "DIMON_training_data_healthy.npz")
        dimon_data = np.load(dimon_path)
        coords = dimon_data['cartesian_coords'].astype(np.float32)  # (50797, 3)
        print(f"Trunk: 3D Cartesian xyz from {dimon_path}")
    else:
        coords = cobiveco_coords                                   # (50797, 4)
        print("Trunk: 4D Cobiveco (ab, rt, tm, tv)")

    # Subsample time if needed
    vm_all = vm_all[:, :, ::frame_step]
    time_ms = time_ms[::frame_step]

    n_total, n_pts, n_time = vm_all.shape
    print(f"Data: {n_total} cases, {n_pts} nodes, {n_time} time steps")
    print(f"Time range: [{time_ms[0]:.0f}, {time_ms[-1]:.0f}] ms, step={time_ms[1]-time_ms[0]:.0f} ms")

    # Split
    num_train = 95
    num_val = 5
    num_test = n_total - num_train - num_val
    print(f"Split: {num_train} train / {num_val} val / {num_test} test")

    f_train = theta[:num_train]
    vm_train_raw = vm_all[:num_train]
    f_val = theta[num_train:num_train + num_val]
    vm_val_raw = vm_all[num_train:num_train + num_val]
    f_test = theta[num_train + num_val:]
    vm_test_raw = vm_all[num_train + num_val:]

    ## Normalization
    # Trunk spatial: per-feature min-max (same recipe for Cobiveco and xyz)
    coords_t = torch.tensor(coords, dtype=torch.float).to(device)
    x_min = coords_t.min(dim=0, keepdim=True)[0]
    x_max = coords_t.max(dim=0, keepdim=True)[0]
    coords_norm = (coords_t - x_min) / (x_max - x_min + 1e-8)

    # Time: normalize to [0, 1]
    t_min, t_max = time_ms.min(), time_ms.max()
    time_norm = (time_ms - t_min) / (t_max - t_min + 1e-8)
    time_t = torch.tensor(time_norm, dtype=torch.float).to(device)

    # Geometry: z-score
    f_mean, f_std = f_train.mean(axis=0), f_train.std(axis=0)
    f_std[f_std < 1e-8] = 1.0
    f_train_norm = (f_train - f_mean) / f_std
    f_val_norm = (f_val - f_mean) / f_std
    f_test_norm = (f_test - f_mean) / f_std

    # V_m: min-shift + std-scale (training stats only)
    vm_mean = vm_train_raw.min()
    vm_std = vm_train_raw.std()
    vm_train_norm = (vm_train_raw - vm_mean) / vm_std
    vm_val_norm = (vm_val_raw - vm_mean) / vm_std
    vm_test_norm = (vm_test_raw - vm_mean) / vm_std

    ## Tensors
    f_train_tensor = torch.tensor(f_train_norm, dtype=torch.float).to(device)
    f_val_tensor = torch.tensor(f_val_norm, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test_norm, dtype=torch.float).to(device)

    # V_m targets: 95 × 50797 × 121 × 4 bytes ≈ 2.3 GB — fits on A100 40GB
    vm_train_tensor = torch.tensor(vm_train_norm, dtype=torch.float).to(device)
    vm_val_tensor = torch.tensor(vm_val_norm, dtype=torch.float).to(device)
    vm_test_tensor = torch.tensor(vm_test_norm, dtype=torch.float).to(device)

    # Precompute trunk chunks once (avoids rebuilding every forward pass)
    print("Precomputing trunk chunks...", flush=True)
    trunk_chunks = precompute_trunk_chunks(coords_norm, time_t,
                                           time_chunk_size, device)
    print(f"  {len(trunk_chunks)} chunks, {sum(c.shape[0] for c in trunk_chunks)} total trunk points")

    model = opnn(dim_br_geo, dim_tr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.lr_schedule:
        from torch.optim.lr_scheduler import LinearLR
        # 0.001 → 0.0001 linearly over the full training run
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                             total_iters=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: geo {dim_br_geo}, trunk {dim_tr}, params {n_params:,}")
    print(f"Temporal chunks: {time_chunk_size} frames/chunk")

    if test_model == 0:
        print(f"--- Starting Training ---", flush=True)

        # Signed-Laplacian neighbour-difference loss operator (row-normalised
        # adjacency on the verified canonical-order mesh). Loss = mean((L E)^2),
        # E = pred - target, L = I - D^-1 A  ==  match Vm - mean(neighbours).
        ndiff_weight = args.ndiff_weight
        Ahat = None
        if ndiff_weight > 0:
            Ahat = build_laplacian_operator(args.adj_file, n_pts, device)
            print(f"ndiff loss ON: signed-Laplacian, weight={ndiff_weight:g}",
                  flush=True)
        else:
            print("ndiff loss OFF (--ndiff-weight 0): baseline MSE", flush=True)

        train_loss_his, val_loss_his, test_loss_his = [], [], []
        ndiff_loss_his = []
        best_val_loss = float('inf')

        # Simple index-based batching (no DataLoader to control memory)
        train_indices = np.arange(num_train)

        tic = timer.time()
        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_indices)
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_ndiff = 0.0
            n_batches = 0

            for b_start in range(0, num_train, batch_size):
                b_idx = train_indices[b_start:b_start + batch_size]
                b_f = f_train_tensor[b_idx]
                b_vm = vm_train_tensor[b_idx]  # already on GPU

                optimizer.zero_grad()
                pred = chunked_forward(model, b_f, trunk_chunks, n_pts)
                err = pred - b_vm                          # (B, M, T)
                mse = (err ** 2).mean()
                if Ahat is not None:
                    ndiff = signed_laplacian_loss(err, Ahat)
                    loss = mse + ndiff_weight * ndiff
                    epoch_ndiff += ndiff.item()
                else:
                    loss = mse
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            avg_mse = epoch_mse / n_batches
            avg_ndiff = epoch_ndiff / n_batches
            train_loss_his.append(avg_train_loss)
            ndiff_loss_his.append(avg_ndiff)

            # Validation + Test (test is held out — monitored, never used for
            # model selection)
            model.eval()
            with torch.no_grad():
                pred_val = chunked_forward(model, f_val_tensor, trunk_chunks, n_pts)
                val_loss = ((pred_val - vm_val_tensor) ** 2).mean().item()
                val_loss_his.append(val_loss)

                pred_test = chunked_forward(model, f_test_tensor, trunk_chunks, n_pts)
                test_loss = ((pred_test - vm_test_tensor) ** 2).mean().item()
                test_loss_his.append(test_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, model_path)

            if args.lr_schedule:
                scheduler.step()

            if epoch % 10 == 0:
                elapsed = timer.time() - tic
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                print(f'Epoch: {epoch}/{epochs} | Train: {avg_train_loss:.6f} | '
                      f'MSE: {avg_mse:.6f} | nDiff: {avg_ndiff:.6f} | '
                      f'Val: {val_loss:.6f} | Test: {test_loss:.6f} | '
                      f'Best Val: {best_val_loss:.6f} | '
                      f'ETA: {int(eta//60)}m{int(eta%60)}s',
                      flush=True)

        total_min = int((timer.time() - tic) / 60)
        print(f"Total training time: {total_min} min")

        np.savetxt(f'./Predictions/{save_directory}/train_loss.txt',
                   np.array(train_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/val_loss.txt',
                   np.array(val_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/test_loss.txt',
                   np.array(test_loss_his))
        np.savetxt(f'./Predictions/{save_directory}/ndiff_loss.txt',
                   np.array(ndiff_loss_his))

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_loss_his, label='Train', alpha=0.8)
        ax.semilogy(val_loss_his, label='Val', alpha=0.8)
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

        with torch.no_grad():
            # Predict test set in single-heart chunks to save memory
            # Warm up GPU (first call has overhead)
            _ = chunked_forward(model, f_test_tensor[:1], trunk_chunks, n_pts)
            torch.cuda.synchronize()

            all_pred = []
            infer_times = []
            for i in range(num_test):
                f_i = f_test_tensor[i:i + 1]
                torch.cuda.synchronize()
                t0 = timer.time()
                pred_i = chunked_forward(model, f_i, trunk_chunks, n_pts)
                torch.cuda.synchronize()
                t1 = timer.time()
                infer_times.append(t1 - t0)
                all_pred.append(to_numpy(pred_i[0]))  # (M, T)
            pred_test = np.stack(all_pred)  # (num_test, M, T)

            print(f"Inference time per case: {np.mean(infer_times)*1000:.1f} +/- "
                  f"{np.std(infer_times)*1000:.1f} ms "
                  f"(total {np.sum(infer_times):.2f} s for {num_test} cases)")

            # Denormalize
            pred_phy = pred_test * vm_std + vm_mean

            # Per-case error metrics
            print(f"\n{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>10}")
            print("-" * 58)
            l2_errors, mae_errors = [], []
            for i in range(num_test):
                u_p = pred_phy[i]       # (M, T)
                u_t = vm_test_raw[i]    # (M, T)
                l2 = np.linalg.norm(u_p - u_t) / np.linalg.norm(u_t)
                mae = np.mean(np.abs(u_p - u_t))
                l2_errors.append(l2)
                mae_errors.append(mae)
                print(f"{case_names[num_train + num_val + i]:<35}"
                      f" {l2:10.4f} {mae:10.2f}")

            print("-" * 58)
            print(f"{'Mean':<35} {np.mean(l2_errors):10.4f} "
                  f"{np.mean(mae_errors):10.2f}")
            print(f"{'Std':<35} {np.std(l2_errors):10.4f} "
                  f"{np.std(mae_errors):10.2f}")

            # Plot V_m traces at a few spatial points for first 2 test hearts.
            # cartesian_coords is only used by the 3D snapshot/AT-scatter pool
            # below (gated by --skip-snapshots), so skip the load if those are
            # off — this lets eval run on Vanda where DIMON_training_data_healthy
            # isn't staged yet.
            if args.skip_snapshots:
                cartesian_coords = None
            else:
                dimon_data = np.load(os.path.join(
                    os.path.dirname(__file__), "..",
                    "DIMON_training_data_healthy.npz"))
                cartesian_coords = dimon_data['cartesian_coords']

            num_viz = min(2, num_test)
            sample_nodes = np.linspace(0, n_pts - 1, 5, dtype=int)  # 5 nodes
            time_axis = time_ms  # physical ms

            # Shared y-range across GT + Pred so the 3 variants line up.
            vm_trace_min = float(min(vm_test_raw[:num_viz, sample_nodes, :].min(),
                                      pred_phy[:num_viz, sample_nodes, :].min()))
            vm_trace_max = float(max(vm_test_raw[:num_viz, sample_nodes, :].max(),
                                      pred_phy[:num_viz, sample_nodes, :].max()))

            def _draw_trace_panels(h, traces, out_path):
                """traces: list of (color, linestyle, lw, array (nodes,T))."""
                fig, axes = plt.subplots(len(sample_nodes), 1,
                                         figsize=(10, 2.5 * len(sample_nodes)),
                                         sharex=True)
                fig.patch.set_alpha(0.0)
                for j, node in enumerate(sample_nodes):
                    ax = axes[j]
                    ax.patch.set_alpha(0.0)
                    for color, ls, lw, arr in traces:
                        ax.plot(time_axis, arr[node, :], color=color,
                                linestyle=ls, linewidth=lw)
                    ax.set_ylim(vm_trace_min, vm_trace_max)
                    ax.set_yticks([])
                    ax.set_ylabel('')
                axes[-1].set_xlabel('Time (ms)', fontsize=30)
                axes[-1].tick_params(axis='x', labelsize=30)
                plt.tight_layout()
                plt.savefig(out_path, format='svg',
                            bbox_inches='tight', transparent=True)
                plt.close()

            for h in range(num_viz):
                gt = vm_test_raw[h]
                pr = pred_phy[h]
                _draw_trace_panels(
                    h, [('k', '-', 3.0, gt)],
                    os.path.join(dump_test, f'vm_traces_heart{h}_GT.svg'))
                _draw_trace_panels(
                    h, [('r', '-', 3.0, pr)],
                    os.path.join(dump_test, f'vm_traces_heart{h}_Pred.svg'))
                _draw_trace_panels(
                    h,
                    [('k', '-', 3.0, gt),
                     ('r', '--', 2.5, pr)],
                    os.path.join(dump_test, f'vm_traces_heart{h}_combined.svg'))

            # ── Viz setup: fixed color scales; tasks collected then parallelised
            num_viz_hearts = min(2, num_test)
            # Frame selection from --vm-frames "START:END:STEP" (END inclusive, ms)
            try:
                vf_start, vf_end, vf_step = (float(x) for x in args.vm_frames.split(':'))
            except Exception:
                raise SystemExit(f"--vm-frames must be START:END:STEP, got {args.vm_frames!r}")
            viz_t_ms = np.arange(vf_start, vf_end + 0.5 * vf_step, vf_step)
            frame_idx = [int(np.argmin(np.abs(time_ms - t))) for t in viz_t_ms]
            print(f"V_m snapshot frames: {len(viz_t_ms)} @ t = "
                  f"{viz_t_ms[0]:.0f}..{viz_t_ms[-1]:.0f} ms (step {vf_step:.0f})")

            vm_min, vm_max = -90.0, 50.0           # mV
            vm_err_min, vm_err_max = 0.0, 20.0     # mV
            cmap_vm = 'RdYlBu_r'
            cmap_at = 'RdYlBu_r'
            cmap_err = 'Reds'

            plot_tasks = []  # (values, cmap, vmin, vmax, out_path)

            def queue_scatter(values, cmap, vmin, vmax, out_path):
                plot_tasks.append((np.asarray(values, dtype=np.float32),
                                   cmap, float(vmin), float(vmax), out_path))

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

            # ── V_m snapshots (per frame, per heart): 3 plots each ───────────
            if args.skip_snapshots:
                print("Skipping V_m snapshot rendering (--skip-snapshots)")
            else:
                snap_dir = os.path.join(dump_test, "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                print(f"Queuing V_m snapshot tasks -> {snap_dir}")

                for h in range(num_viz_hearts):
                    for fi in frame_idx:
                        t_val = time_ms[fi]
                        u_p = pred_phy[h, :, fi]
                        u_t = vm_test_raw[h, :, fi]
                        err = np.abs(u_t - u_p)
                        tag = f"heart{h}_t{t_val:03.0f}ms"
                        queue_scatter(u_t, cmap_vm, vm_min, vm_max,
                                      os.path.join(snap_dir, f"{tag}_GT.svg"))
                        queue_scatter(u_p, cmap_vm, vm_min, vm_max,
                                      os.path.join(snap_dir, f"{tag}_Pred.svg"))
                        queue_scatter(err, cmap_err, vm_err_min, vm_err_max,
                                      os.path.join(snap_dir, f"{tag}_AbsErr.svg"))

            # ── Activation-time maps (first upward crossing of -10 mV) ───────
            def compute_at(vm, t_ms, threshold=-10.0):
                """vm (M, T); returns (M,) AT in ms with linear interpolation.
                NaN for nodes that never cross the threshold."""
                crossed = (vm[:, :-1] < threshold) & (vm[:, 1:] >= threshold)
                any_cross = crossed.any(axis=1)
                first_idx = np.argmax(crossed, axis=1)
                at = np.full(vm.shape[0], np.nan, dtype=np.float32)
                m_idx = np.where(any_cross)[0]
                i = first_idx[m_idx]
                v0 = vm[m_idx, i]
                v1 = vm[m_idx, i + 1]
                t0 = t_ms[i]
                t1 = t_ms[i + 1]
                frac = (threshold - v0) / (v1 - v0 + 1e-12)
                at[m_idx] = t0 + frac * (t1 - t0)
                return at

            at_dir = os.path.join(dump_test, "activation_time")
            os.makedirs(at_dir, exist_ok=True)

            # Compute AT for ALL test hearts (not only viz ones) for metrics
            at_pred_all = np.stack([compute_at(pred_phy[h], time_ms)
                                    for h in range(num_test)])
            at_true_all = np.stack([compute_at(vm_test_raw[h], time_ms)
                                    for h in range(num_test)])

            # Per-case AT MAE and Rel L2 (over nodes that activated in both)
            print(f"\n{'AT metrics':<35} {'Rel L2':>10} {'MAE (ms)':>10}")
            print("-" * 58)
            at_l2_list, at_mae_list = [], []
            for h in range(num_test):
                ap, at = at_pred_all[h], at_true_all[h]
                valid = np.isfinite(ap) & np.isfinite(at)
                diff = ap[valid] - at[valid]
                l2 = np.linalg.norm(diff) / (np.linalg.norm(at[valid]) + 1e-12)
                mae = float(np.mean(np.abs(diff))) if diff.size else float('nan')
                at_l2_list.append(float(l2))
                at_mae_list.append(mae)
                print(f"{case_names[num_train + num_val + h]:<35}"
                      f" {l2:10.4f} {mae:10.2f}")
            print("-" * 58)
            print(f"{'Mean':<35} {np.mean(at_l2_list):10.4f} "
                  f"{np.mean(at_mae_list):10.2f}")
            print(f"{'Std':<35} {np.std(at_l2_list):10.4f} "
                  f"{np.std(at_mae_list):10.2f}")

            # Global AT color range (over viz hearts) shared with standalone cbar
            at_viz_stack = np.concatenate(
                [at_pred_all[h] for h in range(num_viz_hearts)] +
                [at_true_all[h] for h in range(num_viz_hearts)])
            at_vmin = float(np.nanmin(at_viz_stack))
            at_vmax = float(np.nanmax(at_viz_stack))
            at_err_stack = np.concatenate([
                np.abs(at_pred_all[h] - at_true_all[h])
                for h in range(num_viz_hearts)])
            at_err_vmax = float(np.nanmax(at_err_stack)) if at_err_stack.size else 1.0

            if args.skip_snapshots:
                print("Skipping AT scatter rendering (--skip-snapshots)")
            else:
                print(f"Queuing AT tasks -> {at_dir}")
                for h in range(num_viz_hearts):
                    at_pred = at_pred_all[h]
                    at_true = at_true_all[h]
                    err = np.abs(at_true - at_pred)
                    tag = f"heart{h}"
                    queue_scatter(at_true, cmap_at, at_vmin, at_vmax,
                                  os.path.join(at_dir, f"{tag}_AT_GT.svg"))
                    queue_scatter(at_pred, cmap_at, at_vmin, at_vmax,
                                  os.path.join(at_dir, f"{tag}_AT_Pred.svg"))
                    queue_scatter(err, cmap_err, 0.0, at_err_vmax,
                                  os.path.join(at_dir, f"{tag}_AT_AbsErr.svg"))

            # ── Run all scatter tasks in parallel ─────────────────────────────
            if plot_tasks:
                n_workers = min(8, (os.cpu_count() or 4))
                print(f"Rendering {len(plot_tasks)} SVG scatters on "
                      f"{n_workers} processes ...", flush=True)
                t0_render = timer.time()
                with ProcessPoolExecutor(max_workers=n_workers,
                                         initializer=_init_plot_worker,
                                         initargs=(cartesian_coords,)) as ex:
                    list(ex.map(_render_scatter_svg, plot_tasks))
                print(f"  done in {timer.time() - t0_render:.1f} s")

            # ── Standalone colorbars (shared across all above plots) ─────────
            cbar_dir = os.path.join(dump_test, "colorbars")
            os.makedirs(cbar_dir, exist_ok=True)
            save_colorbar(cmap_vm, vm_min, vm_max, 'V_m (mV)',
                          os.path.join(cbar_dir, 'cbar_Vm.svg'))
            save_colorbar(cmap_err, vm_err_min, vm_err_max, '|ΔV_m| (mV)',
                          os.path.join(cbar_dir, 'cbar_Vm_AbsErr.svg'),
                          half=True)
            save_colorbar(cmap_at, at_vmin, at_vmax, 'AT (ms)',
                          os.path.join(cbar_dir, 'cbar_AT.svg'))
            save_colorbar(cmap_err, 0.0, at_err_vmax, '|ΔAT| (ms)',
                          os.path.join(cbar_dir, 'cbar_AT_AbsErr.svg'),
                          half=True)
            print(f"Saved colorbars to {cbar_dir}")

            # Save predictions
            np.savez_compressed(
                os.path.join(dump_test, "test_predictions.npz"),
                pred=pred_phy, true=vm_test_raw,
                at_pred=at_pred_all, at_true=at_true_all,
                at_rel_l2=np.array(at_l2_list), at_mae=np.array(at_mae_list),
                case_names=case_names[num_train + num_val:])
            print(f"\nEvaluation complete. Plots in {dump_test}")


if __name__ == "__main__":
    main()
