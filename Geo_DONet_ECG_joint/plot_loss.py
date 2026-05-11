"""
Quick one-shot script to plot existing train/val loss for Geo_DONet_ECG.
Run from DIMON/Geo_DONet_ECG/:
    source ~/load_dimon_env.sh
    python plot_loss.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

PRED_DIR = "Predictions/geo_donet_ecg_5000ep_0.0005lr_lambda1.0"

train = np.loadtxt(os.path.join(PRED_DIR, "train_loss.txt"))
val   = np.loadtxt(os.path.join(PRED_DIR, "val_loss.txt"))

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(train, label="Train (VM + ECG)", alpha=0.8)
ax.semilogy(val,   label="Val   (VM + ECG)", alpha=0.8)
ax.axvline(np.argmin(val), color="r", ls="--", lw=0.8,
           label=f"Best val @ ep {np.argmin(val)} = {val.min():.5f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("geo_donet_ecg_5000ep_0.0005lr_lambda1.0")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = os.path.join(PRED_DIR, "loss_curve.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")
print(f"  Train final : {train[-1]:.6f}  |  min: {train.min():.6f} @ ep {np.argmin(train)}")
print(f"  Val   final : {val[-1]:.6f}  |  min: {val.min():.6f} @ ep {np.argmin(val)}")
