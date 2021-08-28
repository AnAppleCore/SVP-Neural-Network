import torch
import torch.nn as nn

from models import get_backend
from .voneblock import get_voneblock
from .vtwoblock import get_vtwoblock

class SVPNN(nn.Module):

    def __init__(self, model_arch = 'densenet121'):
        super(SVPNN, self).__init__()

        self.model_arch = model_arch

        self.voneblock = get_voneblock()
        self.vtwoblock = get_vtwoblock(self.model_arch)
        self.backend = get_backend(self.model_arch)

    def forward(self, x):
        y = self.voneblock(x)
        y = self.vtwoblock(y)
        return self.backend(y)