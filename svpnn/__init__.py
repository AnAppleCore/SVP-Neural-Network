
import torch
import torch.nn as nn
from .svpnn import SVPNN

def get_svpnn(model_arch = 'densenet121'):

    print('Model: S{}'.format(model_arch))
    model = SVPNN(model_arch)

    model = nn.DataParallel(model)

    return model

def svpnn_test():
    svpnn = get_svpnn(model_arch = 'densenet121')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    svpnn.to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    y = svpnn(x)
    print(y.shape)