"""Train and evaluate a vanilla pointwise MLP against the Geo-DONet benchmark.

The model directly learns

    f(theta, cobiveco, time) -> V_m

with one shared Tanh MLP. It has no DeepONet branch, trunk, latent dot product,
or separability constraint. Training uses uniformly sampled node/time queries;
full held-out fields are reconstructed in bounded query chunks at test time.
"""

import argparse
import os
import sys
import time as timer
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
GEODONET_DIR = PROJECT_ROOT / "Geo_DONet"
sys.path.insert(0, str(GEODONET_DIR))

from utils import (  # noqa: E402
    Normalizer,
    activation_time,
    at_rel_l2_mae,
    load_dataset,
    split_indices,
    vm_rel_l2_mae,
)
from viz import (  # noqa: E402
    format_metric_summary,
    format_metric_table,
    save_vm_traces,
    write_vtu_series,
)
from model import (  # noqa: E402
    GeoMLP,
    build_from_checkpoint_config,
    config_from_state_dict,
    paired_forward,
)


DATA_FILE = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
REFERENCE_VTU = "/home/svu/e1032484/scratch/canonical.vtu"
N_TRAIN, N_VAL = 95, 5
WIDTH, DEPTH = 300, 4
EPOCHS, CASE_BATCH_SIZE = 5000, 8
SAMPLES_PER_CASE = 4096
VAL_SAMPLES_PER_CASE = 16384
LR, PATIENCE, SEED = 5e-4, 500, 42
FRAME_STEP = 5
QUERY_BATCH_SIZE = 131072
QUERY_FRAME_CHUNK = 25
AT_THRESHOLD_MV = -10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vanilla MLP: [geometry, coordinates, time] -> V_m"
    )
    parser.add_argument("--test-model", action="store_true",
                        help="evaluate a checkpoint on the held-out test split")
    parser.add_argument("--data-path", default=DATA_FILE)
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP,
                        help="time stride applied to f601; 5 gives f121")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--case-batch-size", type=int, default=CASE_BATCH_SIZE,
                        help="number of hearts per optimizer batch")
    parser.add_argument("--samples-per-case", type=int, default=SAMPLES_PER_CASE,
                        help="random (node,time) samples per heart in each batch")
    parser.add_argument("--val-samples-per-case", type=int,
                        default=VAL_SAMPLES_PER_CASE,
                        help="fixed query sample used for validation/early stopping")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr-scheduled", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None,
                        help="cuda/cpu; default chooses CUDA when available")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--query-batch-size", type=int, default=QUERY_BATCH_SIZE,
                        help="maximum pointwise MLP rows per inference call")
    parser.add_argument("--query-frame-chunk", type=int,
                        default=QUERY_FRAME_CHUNK,
                        help="number of output frames assembled at once")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="save test predictions.npz")
    parser.add_argument("--snapshot", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="save trace plots for the first test heart")
    parser.add_argument("--vtu-out", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="save first test heart as a VTU/PVD series")
    parser.add_argument("--mesh", default=REFERENCE_VTU)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def validate_args(args):
    positive = {
        "--frame-step": args.frame_step,
        "--case-batch-size": args.case_batch_size,
        "--samples-per-case": args.samples_per_case,
        "--val-samples-per-case": args.val_samples_per_case,
        "--width": args.width,
        "--depth": args.depth,
        "--query-batch-size": args.query_batch_size,
        "--query-frame-chunk": args.query_frame_chunk,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise SystemExit(f"arguments must be positive: {', '.join(invalid)}")
    if args.n_train < 1 or args.n_val < 1:
        raise SystemExit("--n-train and --n-val must both be positive")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_strided_data(path, frame_step):
    if not os.path.exists(path):
        raise SystemExit(f"data not found: {path}")
    data = load_dataset(path)
    if frame_step > 1:
        # Make compact arrays rather than retaining a strided view whose base is
        # the full f601 allocation.
        data["time"] = data["time"][::frame_step].copy()
        data["vm"] = data["vm"][..., ::frame_step].copy()
    return data


def checkpoint_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def default_checkpoint_name(args, n_frames):
    name = (
        f"geomlp_w{args.width}_d{args.depth}_{args.epochs}ep_"
        f"s{args.samples_per_case}_f{n_frames}"
    )
    if args.lr_scheduled:
        name += "_lrsched"
    if args.seed != SEED:
        name += f"_seed{args.seed}"
    return os.path.join("CheckPts", name + ".pt")


def make_query(coords, time, node_index, time_index):
    return torch.cat(
        (coords[node_index], time[time_index, None]), dim=1
    )


def sampled_targets(vm, case_index, node_index, time_index):
    return vm[
        case_index[:, None], node_index[None, :], time_index[None, :]
    ]


def train(args, device):
    data = load_strided_data(args.data_path, args.frame_step)
    theta, coords, vm, time_ms = (
        data["theta"], data["coords"], data["vm"], data["time"]
    )
    n_cases, n_nodes, n_frames = vm.shape
    train_idx, val_idx, test_idx = split_indices(
        n_cases, args.n_train, args.n_val
    )
    if not len(val_idx):
        raise SystemExit("validation split is empty")
    print(
        f"data: {args.data_path} (frame-step {args.frame_step})\n"
        f"  {n_cases} cases, {n_nodes} nodes, {n_frames} frames | "
        f"{len(train_idx)} train / {len(val_idx)} val / "
        f"{len(test_idx)} test held out"
    )

    normalizer = Normalizer(theta[train_idx], vm[train_idx], coords, time_ms)
    to_device = lambda values: torch.as_tensor(
        values, dtype=torch.float32, device=device
    )
    theta_train = to_device(normalizer.theta(theta[train_idx]))
    theta_val = to_device(normalizer.theta(theta[val_idx]))
    vm_train = to_device(normalizer.vm(vm[train_idx]))
    vm_val = to_device(normalizer.vm(vm[val_idx]))
    coords_norm = to_device(normalizer.coords(coords))
    time_norm = to_device(normalizer.time(time_ms))

    model = GeoMLP(
        geo_dim=theta.shape[1], coord_dim=coords.shape[1],
        width=args.width, depth=args.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        LinearLR(optimizer, 1.0, 0.1, args.epochs)
        if args.lr_scheduled else None
    )
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model: GeoMLP [{theta.shape[1]} shape + {coords.shape[1]} coords + "
        f"1 time] -> 1 | w{args.width} d{args.depth} | "
        f"{n_parameters:,} params | {device}"
    )
    print(
        f"sampling: {args.samples_per_case} random queries/heart/batch | "
        f"{args.val_samples_per_case} fixed queries/heart for validation"
    )

    checkpoint_path = args.model_path or default_checkpoint_name(args, n_frames)
    experiment = checkpoint_stem(checkpoint_path)
    pred_dir = os.path.join("Predictions", experiment)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"checkpoint -> {checkpoint_path} | logs -> {pred_dir}/")

    # Fixed validation locations make early-stopping changes attributable to the
    # model rather than a different Monte-Carlo sample every epoch.
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1000)
    val_nodes = torch.randint(
        n_nodes, (args.val_samples_per_case,), generator=generator,
        device=device,
    )
    val_times = torch.randint(
        n_frames, (args.val_samples_per_case,), generator=generator,
        device=device,
    )
    val_query = make_query(coords_norm, time_norm, val_nodes, val_times)
    val_cases = torch.arange(len(val_idx), device=device)
    val_target = sampled_targets(vm_val, val_cases, val_nodes, val_times)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    order = np.arange(len(train_idx))
    start_time = timer.time()
    loss_path = os.path.join(pred_dir, "loss.txt")
    with open(loss_path, "w") as loss_log:
        loss_log.write("# epoch\ttrain_sampled_mse\tval_fixed_sampled_mse\n")
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(order)
            loss_sum = 0.0
            n_batches = 0
            for start in range(0, len(order), args.case_batch_size):
                batch_np = order[start:start + args.case_batch_size]
                batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
                node_index = torch.randint(
                    n_nodes, (args.samples_per_case,), device=device
                )
                time_index = torch.randint(
                    n_frames, (args.samples_per_case,), device=device
                )
                query = make_query(
                    coords_norm, time_norm, node_index, time_index
                )
                target = sampled_targets(
                    vm_train, batch, node_index, time_index
                )
                prediction = paired_forward(model, theta_train[batch], query)
                loss = torch.mean((prediction - target) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
                n_batches += 1
            if scheduler is not None:
                scheduler.step()
            train_loss = loss_sum / n_batches

            model.eval()
            with torch.no_grad():
                val_prediction = paired_forward(model, theta_val, val_query)
                val_loss = float(
                    torch.mean((val_prediction - val_target) ** 2).item()
                )
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            loss_log.write(f"{epoch}\t{train_loss:.8f}\t{val_loss:.8f}\n")

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "geo_dim": theta.shape[1],
                            "coord_dim": coords.shape[1],
                            "width": args.width,
                            "depth": args.depth,
                        },
                        "training_config": vars(args),
                    },
                    checkpoint_path,
                )
            else:
                stale += 1

            if epoch % 10 == 0:
                loss_log.flush()
                elapsed = timer.time() - start_time
                eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
                print(
                    f"epoch {epoch:5d}/{args.epochs} | "
                    f"train {train_loss:.6f} | val {val_loss:.6f} | "
                    f"best {best_val:.6f} @ {best_epoch} | "
                    f"eta {eta / 60:.1f} min",
                    flush=True,
                )
            if args.patience > 0 and stale >= args.patience:
                print(
                    f"early stop at epoch {epoch}: no validation improvement "
                    f"for {args.patience} epochs",
                    flush=True,
                )
                break

    elapsed = timer.time() - start_time
    print(
        f"done in {elapsed / 60:.1f} min | best val {best_val:.6f} "
        f"at epoch {best_epoch}"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(history["train"], label="train sampled MSE")
    axis.semilogy(history["val"], label="val fixed sampled MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized MSE")
    axis.set_title(experiment)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(pred_dir, "loss.png"), dpi=150)
    plt.close(figure)


def load_model(path, geo_dim, coord_dim, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw)
    inferred = config_from_state_dict(state)
    model = build_from_checkpoint_config(
        inferred, geo_dim=geo_dim, coord_dim=coord_dim
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, inferred


def full_field_forward(
    model, theta_case, coords, time, query_batch_size, frame_chunk
):
    """One case -> (nodes, frames), bounded by query_batch_size rows."""
    n_nodes, coord_dim = coords.shape
    frame_outputs = []
    for frame_start in range(0, len(time), frame_chunk):
        frame_end = min(frame_start + frame_chunk, len(time))
        count = frame_end - frame_start
        coords_tiled = coords[:, None, :].expand(
            n_nodes, count, coord_dim
        ).reshape(n_nodes * count, coord_dim)
        time_tiled = time[frame_start:frame_end][None, :].expand(
            n_nodes, count
        ).reshape(n_nodes * count, 1)
        query = torch.cat((coords_tiled, time_tiled), dim=1)
        pieces = []
        for query_start in range(0, len(query), query_batch_size):
            query_end = min(query_start + query_batch_size, len(query))
            pieces.append(
                paired_forward(
                    model, theta_case[None, :], query[query_start:query_end]
                )[0]
            )
        chunk = torch.cat(pieces).reshape(n_nodes, count)
        frame_outputs.append(chunk.cpu())
    return torch.cat(frame_outputs, dim=1).numpy()


def test(args, device):
    if not args.model_path:
        raise SystemExit("--test-model requires --model-path CHECKPOINT")
    data = load_strided_data(args.data_path, args.frame_step)
    theta, coords, vm, time_ms = (
        data["theta"], data["coords"], data["vm"], data["time"]
    )
    n_cases, n_nodes, n_frames = vm.shape
    train_idx, val_idx, test_idx = split_indices(
        n_cases, args.n_train, args.n_val
    )
    if not len(test_idx):
        raise SystemExit("test split is empty")
    normalizer = Normalizer(theta[train_idx], vm[train_idx], coords, time_ms)
    model, config = load_model(
        args.model_path, theta.shape[1], coords.shape[1], device
    )
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model: GeoMLP {config} | {n_parameters:,} params | {device}")
    print(
        f"test: {len(test_idx)} hearts, {n_nodes} nodes, {n_frames} frames | "
        f"query batch {args.query_batch_size}"
    )

    to_device = lambda values: torch.as_tensor(
        values, dtype=torch.float32, device=device
    )
    theta_test = to_device(normalizer.theta(theta[test_idx]))
    coords_norm = to_device(normalizer.coords(coords))
    time_norm = to_device(normalizer.time(time_ms))
    is_cuda = str(device).startswith("cuda")
    predictions = []
    inference_times = []
    with torch.no_grad():
        for position in range(len(test_idx)):
            if is_cuda:
                torch.cuda.synchronize()
            start = timer.time()
            normalized = full_field_forward(
                model,
                theta_test[position],
                coords_norm,
                time_norm,
                args.query_batch_size,
                args.query_frame_chunk,
            )
            if is_cuda:
                torch.cuda.synchronize()
            inference_times.append(timer.time() - start)
            predictions.append(normalized)
            print(
                f"case {position + 1:2d}/{len(test_idx)} "
                f"({int(test_idx[position])}) | "
                f"{inference_times[-1]:.2f} s",
                flush=True,
            )
    prediction = normalizer.vm_inverse(np.stack(predictions)).astype(np.float32)
    truth = vm[test_idx]
    inference_times = np.asarray(inference_times)

    vm_rel, vm_mae = vm_rel_l2_mae(prediction, truth)
    at_pred = np.stack([
        activation_time(values, time_ms, AT_THRESHOLD_MV)
        for values in prediction
    ])
    at_true = np.stack([
        activation_time(values, time_ms, AT_THRESHOLD_MV)
        for values in truth
    ])
    at_rel, at_mae = at_rel_l2_mae(at_pred, at_true)
    case_names = (
        data["case_names"][test_idx]
        if data["case_names"] is not None else None
    )
    print(
        f"inference: {inference_times.mean():.2f} +/- "
        f"{inference_times.std():.2f} s/case"
    )
    print(format_metric_table(
        "V_m", case_names, vm_rel, vm_mae, "Rel L2", "MAE (mV)"
    ))
    print(format_metric_table(
        "AT", case_names, at_rel, at_mae, "Rel L2", "MAE (ms)"
    ))
    print(format_metric_summary(
        len(test_idx), "test", vm_rel, vm_mae, at_rel, at_mae
    ))

    out_dir = args.out_dir or os.path.join(
        "Predictions", checkpoint_stem(args.model_path), "Test"
    )
    os.makedirs(out_dir, exist_ok=True)
    labels = [
        str(case_names[i]) if case_names is not None
        else f"case{int(test_idx[i])}"
        for i in range(len(test_idx))
    ]
    with open(os.path.join(out_dir, "test_log.txt"), "w") as handle:
        handle.write(format_metric_summary(
            len(test_idx), "test", vm_rel, vm_mae, at_rel, at_mae
        ) + "\n")
    if args.snapshot:
        save_vm_traces(
            prediction[0], truth[0], time_ms,
            os.path.join(out_dir, "first_case_traces.png"),
        )
    if args.vtu_out:
        if not os.path.exists(args.mesh):
            raise SystemExit(f"mesh not found: {args.mesh}")
        mesh = meshio.read(args.mesh)
        tetra = next(
            (block.data for block in mesh.cells if block.type == "tetra"), None
        )
        if tetra is None or len(mesh.points) != n_nodes:
            raise SystemExit("canonical mesh is inconsistent with prediction nodes")
        fields = {
            "Vm": prediction[0],
            "Vm_gt": truth[0],
            "Vm_abserr": np.abs(prediction[0] - truth[0]),
        }
        write_vtu_series(
            os.path.join(out_dir, "vtu", labels[0]),
            mesh.points.astype(np.float32),
            tetra.astype(np.int32),
            time_ms,
            fields,
        )
    if args.save:
        np.savez_compressed(
            os.path.join(out_dir, "predictions.npz"),
            pred=prediction,
            true=truth,
            time=time_ms,
            case_names=np.asarray(labels),
            vm_rel_l2=vm_rel,
            vm_mae=vm_mae,
            at_pred=at_pred,
            at_true=at_true,
            at_rel_l2=at_rel,
            at_mae=at_mae,
            infer_time_per_case_s=inference_times,
        )
    print(f"outputs -> {out_dir}")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.test_model:
        test(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
