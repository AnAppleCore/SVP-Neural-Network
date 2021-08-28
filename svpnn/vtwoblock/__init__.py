
import torch
from .vtwoblock import VTwoBlock

def get_vtwoblock(model_arch = 'densenet'):

    if model_arch == 'alexnet':
        out_channels = 192
    else:
        out_channels = 128

    vtwoblock = VTwoBlock(out_channels)
    print('V2: AlexNet pre-trained layer 2')

    return vtwoblock


def vtwo_test():
    vtwo = get_vtwoblock(model_arch = 'alexnet')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    vtwo.to(device)
    x = torch.randn(4, 64, 56, 56).to(device)
    y = vtwo(x)
    print(y.shape)
