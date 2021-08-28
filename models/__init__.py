import os
import torch
import torch.nn as nn

from .alexnet import AlexNet, AlexNetBackEnd
from .vgg19_bn import VGG19_BN, VGGBackEnd
from .densenet121 import DenseNet121, DenseNetBackEnd


AVAILABLE_MODEL = {'densenet121', 'alexnet', 'vgg19_bn'}


def get_ori_model(model_arch = 'densenet121'):
    """
        Return the original neural network models.
        @param:
            'model_arch': {'densenet121', 'alexnet', 'vgg19_bn'}
    """

    assert model_arch in AVAILABLE_MODEL, 'Model Unsupported!'

    if model_arch == 'densenet121':
        model = DenseNet121()
        print('Model: ', 'DenseNet121')
    elif model_arch == 'alexnet':
        model = AlexNet()
        print('Model: ', 'AlexNet')
    elif model_arch == 'vgg19_bn':
        model = VGG19_BN()
        print('Model: ', 'VGG19-BN')

    model = nn.DataParallel(model)

    return model


def model_test(model_arch = 'densenet121'):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = get_ori_model(model_arch=model_arch)
    model.to(device)
    print(model)

    x = torch.randn(4, 3, 224, 224).to(device)
    y = model(x).to('cpu')
    x = x.to('cpu')
    print(x.shape, y.shape)


def get_backend(model_arch = 'densenet121'):
    """
        Return the backends.
        @param:
            'model_arch': {'densenet121', 'alexnet', 'vgg19_bn'}
    """

    assert model_arch in AVAILABLE_MODEL, 'Model Unsupported!'

    if model_arch == 'densenet121':
        backend = DenseNetBackEnd()
        print('Backend model: ', 'DenseNet121')
    elif model_arch == 'alexnet':
        backend = AlexNetBackEnd()
        print('Backend model: ', 'AlexNet')
    elif model_arch == 'vgg19_bn':
        backend = VGGBackEnd()
        print('Backend model: ', 'VGG19-BN')

    return backend