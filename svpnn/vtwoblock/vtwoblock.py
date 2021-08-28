import os
import requests
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


vtwo_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(vtwo_dir, 'vtwoweight.pth')
if not os.path.exists(weight_path):
    weights_path =os.path.join(vtwo_dir, 'alexnet-owt-7be5be79.pth')
    if not os.path.exists(weights_path):
        print('Downloading pretrained alexnet weight...')
        url = 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
        r = requests.get(url)
        with open(weights_path, 'wb') as weight:
            weight.write(r.content)
    ckpt_data = OrderedDict()
    alex_data = torch.load(weights_path)
    ckpt_data['weight'] = alex_data['features.3.weight']
    ckpt_data['bias'] = alex_data['features.3.bias']
    torch.save(alex_data, save_path)
else:
    ckpt_data = torch.load(weight_path)


# The second layer of AlexNet, but with pre-trained weights

class VTwoBlock(nn.Module):

    def __init__(self, out_channels = 128):
        super(VTwoBlock, self).__init__()
        self.vtwo = nn.Sequential(
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.bottleneck = bottleneck = nn.Conv2d(192, out_channels, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        self.weight = ckpt_data['weight']
        self.bias = ckpt_data['bias']
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def forward(self, x):
        y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=2)
        y = self.vtwo(y)
        return self.bottleneck(y)