"""
Plot training and test loss curves from saved text files.
Reads from Predictions/<save_directory>/ and saves plot there.

Usage:
    python plot.py
    python plot.py --epochs 1000
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--norm", type=int, default=1)
    args = parser.parse_args()

    save_dir = f"cobiveco_{args.norm}norm_{args.epochs}ep_{args.lr}lr"
    pred_dir = f"./Predictions/{save_dir}"

    train_loss = np.loadtxt(os.path.join(pred_dir, "train_loss.txt"))
    test_loss = np.loadtxt(os.path.join(pred_dir, "test_loss.txt"))

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, color="blue", linewidth=0.8, label="Train MSE")
    ax.plot(epochs, test_loss, color="red", linewidth=0.8, linestyle="--", label="Test MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title(f"Geo-DeepONet Loss — {save_dir}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(pred_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
