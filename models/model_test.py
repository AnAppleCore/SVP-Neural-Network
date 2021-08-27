from alexnet import AlexNet
from vgg19_bn import VGG19_BN
from densenet121 import DenseNet121

import torch

device = torch.device('cuda')
model = DenseNet121()
print(model)
model = model.to(device)

x = torch.randn(4, 3, 224, 224).to(device)
y = model(x).to('cpu')
x = x.to('cpu')
print(x.shape, y.shape)