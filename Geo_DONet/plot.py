"""
Plot training and test loss curves.

Usage:
    python plot.py --dir cobiveco4d_1norm_3000ep_0.0005lr
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_loss(train_path, test_path, save_path):
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Could not find {train_path} or {test_path}")
        return

    # Load the data
    train_loss = np.loadtxt(train_path)
    test_loss = np.loadtxt(test_path)

    # Determine epochs for x-axis
    epochs_train = np.arange(len(train_loss))
    epochs_test = np.arange(len(test_loss))
    
    # Sample test loss every 100 epochs to reduce visual clutter
    sampled_epochs_test = epochs_test[::100]
    sampled_test_loss = test_loss[::100]

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plotting on log scale
    plt.semilogy(epochs_train, train_loss, label='Train Loss', color='blue', alpha=0.7)
    plt.semilogy(sampled_epochs_test, sampled_test_loss, label='Test Loss', 
                 color='red', linestyle='--', marker='o', markersize=4)

    plt.title('Training & Test Loss (Log Scale)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss (Normalized)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save and Show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Predictions subdirectory name")
    args = parser.parse_args()
    d = args.dir if os.path.isabs(args.dir) or args.dir.startswith('./') or args.dir.startswith('../') \
        else f"./Predictions/{args.dir}"
    plot_loss(f"{d}/train_loss.txt", f"{d}/test_loss.txt", f"{d}/loss_curve.png")