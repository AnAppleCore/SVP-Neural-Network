
import torch
import numpy as np

from voneblock import VOneBlock
from params import generate_gabor_param

def get_voneblock(sf_corr=0.75, sf_max=6, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode=None, noise_scale=1, noise_level=1, k_exc=25, image_size=224, visual_degrees=8, ksize=25, stride=4):
    
    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    voneblock = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    return voneblock


def vone_test():
    vone = get_voneblock()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    vone.to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    y = vone(x)
    print(y.shape)


if __name__ == '__main__':
    vone_test()