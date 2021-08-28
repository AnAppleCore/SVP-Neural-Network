import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# The second layer of AlexNet, but with pre-trained weights

class VTwoBlock(nn.Module):

    def __init__(self, weight_path, out_channels = 128):
        super(VTwoBlock, self).__init__()
        self.vtwo = nn.Sequential(
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.bottleneck = bottleneck = nn.Conv2d(192, out_channels, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        ckpt_data = torch.load(weight_path)
        self.weight = ckpt_data['weight']
        self.bias = ckpt_data['bias']
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def forward(self, x):
        y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=2)
        y = self.vtwo(y)
        return self.bottleneck(y)


def extract_v2_weight(save_path = 'vtwoweight.pth'):
    vtwo = OrderedDict()
    #TODO: download the online model here
    weights_path = os.path.join('.', 'alexnet-owt-7be5be79.pth')
    ckpt_data = torch.load(weights_path)
    vtwo['weight'] = ckpt_data['features.3.weight']
    vtwo['bias'] = ckpt_data['features.3.bias']
    torch.save(vtwo, 'vtwoweight.pth')
    return None


def test():
    vtwo = VTwoBlock('vtwoweight.pth')
    for idx, p in enumerate(vtwo.parameters()):
        print(idx, '->', type(p), p.size(), p.requires_grad)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    vtwo.to(device)
    x = torch.randn(4, 64, 128, 128).to(device)
    y = vtwo(x)
    print(y.shape)


if __name__ == '__main__':
    # extract_v2_weight(save_path = 'vtwoweight.pth')
    test()