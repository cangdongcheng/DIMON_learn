"""
ECG Benchmark: direct theta (60D PCA geometry) → ECG (121×10) mapping.

Three architectures:
  mlp  — MLP decoder            60 → [512,512,512] → (T,10)
  gru  — GRU decoder            theta → hidden init + sinusoidal time → (T,10)
  conv — 1D ConvTranspose       theta → (C,8) → upsample → (10,T)

Comparison target from eval_chain.py (95/5/25 split, same data):
  Pipeline A (GT V_m → ECG Transfer)       : 12-lead Rel L2 = 0.241 ± 0.075
  Pipeline B (Geo-DONet V_m → ECG Transfer): 12-lead Rel L2 = 0.303 ± 0.077

Usage:
    source ~/load_dimon_env.sh
    cd DIMON/ECG_benchmark

    python main.py --model mlp  --epochs 5000 --device cuda
    python main.py --model gru  --epochs 5000 --device cuda
    python main.py --model conv --epochs 5000 --device cuda

    python main.py --model mlp  --epochs 5000 --test-model 1 --device cuda
"""
import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"


# ─── Models ──────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """60D → [512,512,512] Tanh → flatten (T×10) → reshape (T,10)."""
    def __init__(self, n_modes=60, n_time=121, n_elec=10,
                 hidden=(512, 512, 512)):
        super().__init__()
        layers, d = [], n_modes
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, n_time * n_elec))
        self.net = nn.Sequential(*layers)
        self.T, self.E = n_time, n_elec

    def forward(self, theta):              # (N,60) → (N,T,10)
        return self.net(theta).view(-1, self.T, self.E)


def sinusoidal_enc(t_norm, n_freq=16, device='cpu'):
    """
    t_norm : (T,) in [0,1]
    returns : (T, 2*n_freq)  — sin + cos at geometric frequencies
    """
    freqs  = 2 ** torch.arange(n_freq, dtype=torch.float, device=device)
    angles = 2 * np.pi * t_norm.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class GRUDecoder(nn.Module):
    """
    theta → Linear → initial hidden state for GRU.
    GRU input at each step: sinusoidal time encoding (T, 32).
    Output projection: Linear → 10 electrodes per step.
    """
    def __init__(self, n_modes=60, n_time=121, n_elec=10,
                 hidden_dim=256, n_layers=2, n_freq=16):
        super().__init__()
        self.T, self.E = n_time, n_elec
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        enc_dim = 2 * n_freq
        self.init_h = nn.Linear(n_modes, hidden_dim * n_layers)
        self.gru    = nn.GRU(enc_dim, hidden_dim, n_layers, batch_first=True)
        self.out    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_elec),
        )

    def forward(self, theta, t_enc):       # theta:(N,60), t_enc:(T,32) → (N,T,10)
        N  = theta.shape[0]
        h0 = (self.init_h(theta)
                  .view(N, self.n_layers, self.hidden_dim)
                  .permute(1, 0, 2).contiguous())      # (layers, N, hidden)
        inp = t_enc.unsqueeze(0).expand(N, -1, -1)     # (N, T, 32)
        out, _ = self.gru(inp, h0)                      # (N, T, hidden)
        return self.out(out)                             # (N, T, 10)


class ConvDecoder(nn.Module):
    """
    Project theta → (N, base_ch, T) directly, then refine with 1D conv along time.
    Avoids ConvTranspose1d upsampling which collapses to zero with Tanh stacking.
    Same family as TemporalConvTransfer (which worked in ECG_transfer).
    """
    def __init__(self, n_modes=60, n_time=121, n_elec=10, base_ch=64):
        super().__init__()
        self.T = n_time
        self.proj = nn.Linear(n_modes, base_ch * n_time)   # project to full T
        self.conv = nn.Sequential(
            nn.Conv1d(base_ch,    base_ch,    5, padding=2), nn.Tanh(),
            nn.Conv1d(base_ch,    base_ch//2, 5, padding=2), nn.Tanh(),
            nn.Conv1d(base_ch//2, n_elec,     5, padding=2),
        )
        self.base_ch = base_ch

    def forward(self, theta):              # (N,60) → (N,T,10)
        N = theta.shape[0]
        x = self.proj(theta).view(N, self.base_ch, self.T)  # (N, C, T)
        x = self.conv(x)                                      # (N, 10, T)
        return x.permute(0, 2, 1)                             # (N, T, 10)


MODEL_MAP = {'mlp': MLPDecoder, 'gru': GRUDecoder, 'conv': ConvDecoder}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def set_seed(s=42):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s);    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def to_numpy(t):
    return t.detach().cpu().numpy()


def to_12lead(raw):
    """(T,10) raw electrode potentials → (T,12) standard leads."""
    LA, RA, LL = raw[:, 0], raw[:, 1], raw[:, 2]
    WCT = (RA + LA + LL) / 3
    return np.column_stack([
        LA - RA, LL - RA, LL - LA,
        RA - (LA + LL) / 2, LA - (RA + LL) / 2, LL - (RA + LA) / 2,
        raw[:, 4] - WCT, raw[:, 5] - WCT, raw[:, 6] - WCT,
        raw[:, 7] - WCT, raw[:, 8] - WCT, raw[:, 9] - WCT,
    ])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str,   default='mlp',
                   choices=['mlp', 'gru', 'conv'])
    p.add_argument('--epochs',     type=int,   default=5000)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--batch-size',    type=int,   default=32)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--device',        type=str,   default='cuda')
    p.add_argument('--test-model',    type=int,   default=0)
    args = p.parse_args()

    device   = args.device
    wd_tag   = f'_wd{args.weight_decay}' if args.weight_decay > 0 else ''
    run_name = f'{args.model}_{args.epochs}ep{wd_tag}'
    save_dir = f'Predictions/{run_name}'
    ckpt     = f'CheckPts/{run_name}.pt'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    data       = np.load(os.path.join(DATA_BASE, "geo_donet_data_f121.npz"),
                         allow_pickle=True)
    theta_all  = data['theta'][:, :60].astype(np.float32)   # (125, 60)
    ecg_all    = data['ecg'].astype(np.float32)              # (125, 121, 10)
    time_ms    = data['time'].astype(np.float32)             # (121,)
    case_names = data['case_names']

    n_total, n_time, n_elec = ecg_all.shape
    print(f"Data: {n_total} cases, {n_time} frames, {n_elec} electrodes")

    num_train, num_val = 95, 5
    num_test = n_total - num_train - num_val

    th_tr  = theta_all[:num_train]
    th_val = theta_all[num_train:num_train + num_val]
    th_te  = theta_all[num_train + num_val:]
    ecg_tr  = ecg_all[:num_train]
    ecg_val = ecg_all[num_train:num_train + num_val]
    ecg_te  = ecg_all[num_train + num_val:]

    print(f"Split: {num_train} train / {num_val} val / {num_test} test")

    # ── Normalisation ────────────────────────────────────────────────────────
    # theta: z-score (per-mode, from training set)
    th_mean, th_std = th_tr.mean(0), th_tr.std(0)
    th_std[th_std < 1e-8] = 1.0
    th_tr_n  = (th_tr  - th_mean) / th_std
    th_val_n = (th_val - th_mean) / th_std
    th_te_n  = (th_te  - th_mean) / th_std

    # ECG: min-shift + std-scale (matches ECG_transfer convention)
    ecg_mean = ecg_tr.min()
    ecg_std  = ecg_tr.std()
    ecg_tr_n  = (ecg_tr  - ecg_mean) / ecg_std
    ecg_val_n = (ecg_val - ecg_mean) / ecg_std

    # ── Tensors ───────────────────────────────────────────────────────────────
    th_tr_t   = torch.tensor(th_tr_n,   dtype=torch.float).to(device)
    th_val_t  = torch.tensor(th_val_n,  dtype=torch.float).to(device)
    th_te_t   = torch.tensor(th_te_n,   dtype=torch.float).to(device)
    ecg_tr_t  = torch.tensor(ecg_tr_n,  dtype=torch.float).to(device)
    ecg_val_t = torch.tensor(ecg_val_n, dtype=torch.float).to(device)

    # Sinusoidal time encoding for GRU (precomputed, reused every step)
    t_norm = torch.tensor(
        (time_ms - time_ms.min()) / (time_ms.max() - time_ms.min()),
        dtype=torch.float).to(device)
    t_enc = sinusoidal_enc(t_norm, n_freq=16, device=device)  # (121, 32)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MODEL_MAP[args.model](n_modes=60, n_time=n_time, n_elec=n_elec).to(device)
    n_params = sum(par.numel() for par in model.parameters())
    print(f"Model: {args.model}, params: {n_params:,}")

    def fwd(theta):
        return model(theta, t_enc) if args.model == 'gru' else model(theta)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # ── Training ──────────────────────────────────────────────────────────────
    if args.test_model == 0:
        train_idx = np.arange(num_train)
        tr_loss_h, val_loss_h = [], []
        best_val = float('inf')
        tic = time.time()
        print(f"--- Training {args.model} ---", flush=True)

        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(train_idx)
            ep_loss, n_b = 0.0, 0

            for s in range(0, num_train, args.batch_size):
                idx  = train_idx[s:s + args.batch_size]
                pred = fwd(th_tr_t[idx])                       # (B, T, 10)
                loss = ((pred - ecg_tr_t[idx]) ** 2).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                ep_loss += loss.item(); n_b += 1

            avg = ep_loss / n_b
            tr_loss_h.append(avg)

            model.eval()
            with torch.no_grad():
                vl = ((fwd(th_val_t) - ecg_val_t) ** 2).mean().item()
            val_loss_h.append(vl)

            if vl < best_val:
                best_val = vl
                torch.save({'model_state_dict': model.state_dict()}, ckpt)

            if epoch % 100 == 0:
                el  = time.time() - tic
                eta = el / (epoch + 1) * (args.epochs - epoch - 1)
                print(f'Epoch {epoch}/{args.epochs} | '
                      f'Train: {avg:.6f} | Val: {vl:.6f} | '
                      f'Best: {best_val:.6f} | '
                      f'ETA: {int(eta//60)}m{int(eta%60)}s', flush=True)

        print(f"Done: {int((time.time()-tic)/60)} min")
        np.savetxt(f'{save_dir}/train_loss.txt', tr_loss_h)
        np.savetxt(f'{save_dir}/val_loss.txt',   val_loss_h)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(tr_loss_h,  label='Train', alpha=0.8)
        ax.semilogy(val_loss_h, label='Val',   alpha=0.8)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title(run_name); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/loss_curve.png', dpi=150)
        plt.close()
        print(f"Loss curve → {save_dir}/loss_curve.png")

    # ── Evaluation ────────────────────────────────────────────────────────────
    else:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        model.eval()

        elec_names = ["LA","RA","LL","RL","V1","V2","V3","V4","V5","V6"]
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

        with torch.no_grad():
            pred_n = to_numpy(fwd(th_te_t))           # (25, T, 10)
        pred_phy = pred_n * ecg_std + ecg_mean

        def print_metrics(label, l2s, maes):
            print(f"{'Mean':<35} {np.mean(l2s):10.4f} {np.mean(maes):12.6f}")
            print(f"{'Std' :<35} {np.std(l2s) :10.4f} {np.std(maes) :12.6f}")
            return np.mean(l2s), np.std(l2s)

        # 10-electrode metrics
        print(f"\n--- 10-Electrode Metrics ({args.model}) ---")
        print(f"{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>12}")
        print("-" * 60)
        l2_10, mae_10 = [], []
        for i in range(num_test):
            p, t = pred_phy[i], ecg_te[i]
            l2  = np.linalg.norm(p - t) / np.linalg.norm(t)
            mae = np.mean(np.abs(p - t))
            l2_10.append(l2); mae_10.append(mae)
            print(f"{case_names[num_train+num_val+i]:<35} {l2:10.4f} {mae:12.6f}")
        print("-" * 60)
        mu_10, sd_10 = print_metrics("10-elec", l2_10, mae_10)

        # 12-lead metrics
        print(f"\n--- 12-Lead Metrics ({args.model}) ---")
        print(f"{'Case':<35} {'Rel L2':>10} {'MAE (mV)':>12}")
        print("-" * 60)
        l2_12, mae_12 = [], []
        for i in range(num_test):
            p12 = to_12lead(pred_phy[i])
            t12 = to_12lead(ecg_te[i])
            l2  = np.linalg.norm(p12 - t12) / np.linalg.norm(t12)
            mae = np.mean(np.abs(p12 - t12))
            l2_12.append(l2); mae_12.append(mae)
            print(f"{case_names[num_train+num_val+i]:<35} {l2:10.4f} {mae:12.6f}")
        print("-" * 60)
        mu_12, sd_12 = print_metrics("12-lead", l2_12, mae_12)

        print(f"\n══ SUMMARY: {args.model}_{args.epochs}ep ══")
        print(f"  10-elec 12-lead Rel L2: {mu_10:.4f} ± {sd_10:.4f}")
        print(f"  12-lead 12-lead Rel L2: {mu_12:.4f} ± {sd_12:.4f}")
        print(f"  (chain ref: A=0.241±0.075  B=0.303±0.077)")

        # ECG trace plots (first 2 test cases)
        num_viz = min(2, num_test)
        for h in range(num_viz):
            fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
            for e in range(10):
                ax = axes[e // 2, e % 2]
                ax.plot(time_ms, ecg_te[h, :, e]*1000, 'k-',  lw=1, label='True')
                ax.plot(time_ms, pred_phy[h, :, e]*1000, 'r--', lw=1, label='Pred')
                ax.set_ylabel(f'{elec_names[e]} (µV)', fontsize=8)
                ax.grid(True, alpha=0.3)
                if e == 0: ax.legend(fontsize=7)
            axes[-1, 0].set_xlabel('Time (ms)')
            axes[-1, 1].set_xlabel('Time (ms)')
            fig.suptitle(
                f'Electrodes — {case_names[num_train+num_val+h]} ({args.model})',
                fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/electrodes_heart{h}.png', dpi=150)
            plt.close()

        np.savez_compressed(f'{save_dir}/predictions.npz',
                            pred_ecg=pred_phy, true_ecg=ecg_te,
                            case_names=case_names[num_train+num_val:])
        print(f"\nPlots → {save_dir}/")


if __name__ == "__main__":
    main()
