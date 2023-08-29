from utils.config import Config
from model.training import train

from process_data.cifar import Cifar10

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', type=str, default='diff_cifar10')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--w_decay', type=float, default=.1)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--lr', type=int, default=3e-4)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n_class', type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # TODO big next steps :
    #   2. Launch it
    #       - Finish training function
    #       - Understand the loss
    #       - Read the paper
    #   3. Test with mitochondria
    #   4. Put it on GitHub

    # Get command line arguments
    args = get_args()

    # Create the dataset
    data = Cifar10(args)

    # Set configuration of the experience
    basedir = 'saved_experiments'
    os.makedirs(basedir, exist_ok=True)

    global_param = {'basedir': basedir, 'dataset': data.name_dataset, 'name_exp': args.name_exp, 'img_size': data.img_size}
    config = Config(args, global_param)

    train(args, data)


