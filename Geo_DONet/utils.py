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
    parser.add_argument('--width', type=int, default=300, metavar='N',
                        help='hidden width for branch and trunk MLPs (default: 300)')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='training batch size (default: 24; reduce to 16 for width=400)')
    parser.add_argument('--lr-schedule', action='store_true',
                        help='linear decay from 0.001 → 0.0005 over first 2500 epochs, then constant')
    parser.add_argument('--model-path', type=str, default=None,
                        help='explicit checkpoint path for --test-model 1 (overrides auto-generated name)')
    parser.add_argument('--skip-snapshots', action='store_true',
                        help='skip the 3D V_m snapshot and AT scatter SVGs (keeps traces + metrics)')
    parser.add_argument('--vm-frames', type=str, default='0:300:10',
                        metavar='START:END:STEP',
                        help='V_m snapshot time range in ms (default: 0:300:10 → '
                             't=0,10,...,300). END is inclusive.')
    parser.add_argument('--trunk', type=str, default='cobiveco',
                        choices=['cobiveco', 'xyz'],
                        help='trunk spatial coordinates: 4D Cobiveco (ab,rt,tm,tv) '
                             'or 3D normalised Cartesian xyz (default: cobiveco)')
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
