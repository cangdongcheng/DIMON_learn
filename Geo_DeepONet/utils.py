"""
Author: Minglang Yin, myin16@jhu.edu
"""
import numpy as np
import torch
import argparse


def ParseArgument():
    parser = argparse.ArgumentParser(description='Geo-DeepONet')
    parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                        help='number of epochs to train (default: 50000)')
    parser.add_argument('--data-path', type=str,
                        default='/home/svu/e1032484/scratch/geo_deeponet_pca_f601_k5.npz',
                        help='AT dataset: legacy key at, or compact targets[...,0]')
    parser.add_argument('--reference-xyz', type=str,
                        default='/home/svu/e1032484/scratch/canonical_xyz.npz',
                        help='cartesian xyz npz used for evaluation plots')
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help='computing device (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--patience', type=int, default=1000,
                        help='early-stop epochs without validation improvement; 0 disables')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-step', type=int, default=10000, metavar='N',
                        help='save_step (default: 10000)')
    parser.add_argument('--test-model', type=int, default=0, metavar='N',
                        help='default training, testing as 1')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='path to checkpoint (.pt) for --test-model 1; '
                             'default: derived from epochs/lr')
    parser.add_argument('--viz-hearts', type=int, default=2,
                        help='number of hearts to plot from train/test (default 2)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='skip 3D scatter rendering (metrics only)')
    args = parser.parse_args()
    return args


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '
                        'np.ndarray, but got {}'.format(type(input)))
