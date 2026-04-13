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
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help='computing device (default: cuda)')
    parser.add_argument('--save-step', type=int, default=10000, metavar='N',
                        help='save_step (default: 10000)')
    parser.add_argument('--test-model', type=int, default=0, metavar='N',
                        help='default training, testing as 1')
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
