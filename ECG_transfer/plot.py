"""Plot training loss curves and ECG comparison for ECG Transfer models."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(model_name, epochs):
    """Plot training and validation loss curves."""
    save_dir = f'Predictions/{model_name}_{epochs}ep'
    train_loss = np.loadtxt(os.path.join(save_dir, 'train_loss.txt'))
    val_loss = np.loadtxt(os.path.join(save_dir, 'val_loss.txt'))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(train_loss, label='Train', alpha=0.8)
    ax.semilogy(val_loss, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'ECG Transfer ({model_name}) — Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_comparison(models, epochs_list):
    """Compare loss curves across model variants on one plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, ep in zip(models, epochs_list):
        save_dir = f'Predictions/{model_name}_{ep}ep'
        if not os.path.exists(os.path.join(save_dir, 'val_loss.txt')):
            print(f"Skipping {model_name} (no results yet)")
            continue
        val_loss = np.loadtxt(os.path.join(save_dir, 'val_loss.txt'))
        ax.semilogy(val_loss, label=f'{model_name}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE Loss')
    ax.set_title('ECG Transfer — Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = 'Predictions/model_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    models = ['linear', 'mlp', 'temporal_conv']
    epochs = [5000, 5000, 5000]

    for m, ep in zip(models, epochs):
        save_dir = f'Predictions/{m}_{ep}ep'
        if os.path.exists(os.path.join(save_dir, 'train_loss.txt')):
            plot_loss(m, ep)

    plot_comparison(models, epochs)
