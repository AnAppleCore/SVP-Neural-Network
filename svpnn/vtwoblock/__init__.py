
import torch
from vtwoblock import VTwoBlock

def get_vtwoblock(model_arch = 'densenet', weight_path = 'vtwoweight.pth'):

    if model_arch == 'alexnet':
        out_channels = 192
    else:
        out_channels = 128

    vtwoblock = VTwoBlock(weight_path, out_channels)

    return vtwoblock


def vtwo_test():
    vtwo = get_vtwoblock(model_arch = 'alexnet')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    vtwo.to(device)
    x = torch.randn(4, 64, 128, 128).to(device)
    y = vtwo(x)
    print(y.shape)


if __name__ == '__main__':
    vtwo_test()