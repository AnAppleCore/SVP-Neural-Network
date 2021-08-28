### Main Script to Run SVPNN

import tqdm
import numpy as np
import pandas as pd
import os, argparse, subprocess, io, shlex

from utils.val import val
from utils.train import train


parser = argparse.ArgumentParser(description='Semi-bionic Visual Pathway Neural Network')

# IO path parameters
parser.add_argument('-i', '--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default='./results/',
                    help='path for storing results')
parser.add_argument('-repoch', '--restore_epoch', default=0, type=int,
                    help='epoch number for restoring model')
parser.add_argument('-rpath', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model')

# Execution parameters
parser.add_argument('-m', '--mode', choices=['train', 'val'], default='val',
                    help='switch to train or validate mode')
parser.add_argument('-g','--ngpus', default=4, type=int,
                    help='number of GPUs to use; 0 for running on CPU')
parser.add_argument('-j', '--workers', default=20, type=int,
                    help='number of data loading workers')
parser.add_argument('-e', '--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('-n', '--no_svp', default=False, action='store_true',
                    help='tigger to run the original model without SVP')
parser.add_argument('-a', '--model_arch', choices=['alexnet', 'vgg19_bn', 'densenet121'], default='densenet121',
                    help='model architecture to run')

# Training and  parameters
parser.add_argument('--optimizer', choices=['stepLR', 'plateauLR'], default='stepLR',
                    help='Optimizer')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epochs learning rate should be decreased by step_factor')
parser.add_argument('--step_factor', default=0.1, type=float,
                    help='factor by which to decrease the learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for randomness')

ARGS = parser.parse_args()


def set_gpus(n=4):
    """
        Finds all at most 'n' available GPUs
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 4096]  # only above 4 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(args = None):

    assert args.mode in {'train', 'val'}

    if args.ngpus > 0:
        set_gpus(args.ngpus)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'val':
        val(args)

    return None


if __name__ == '__main__':
    main(ARGS)