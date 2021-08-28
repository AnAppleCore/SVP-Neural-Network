import torch
import torch.nn as nn
import torchvision

import os, tqdm, time

from models import get_ori_model
from svpnn import get_svpnn

# Global parameters
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]


class ImageNetVal(object):

    def __init__(self, model, device, args):
        self.name = 'val'
        self.model = model
        self.args = args
        self.device = device
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.args.in_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.args.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(self.device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


def val(args = None):

    # initialization
    torch.manual_seed(args.torch_seed)
    torch.backends.cudnn.benchmark = True
    if args.ngpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if args.no_svp:
        print('Running {m} without SVP'.format(m=args.model_arch))
        model = get_ori_model(model_arch=args.model_arch)
    else:
        print('Running SVPNN with {m} backend'.format(m=args.model_arch))
        model = get_svpnn(model_arch = args.model_arch)
    
    if args.ngpus > 0 :
        print('We have {} GPU(s) detected: {}'.format(str(torch.cuda.device_count()), os.environ['CUDA_VISIBLE_DEVICES']))
        model = model.to(device)
    else:
        print('Caution!!! We run on CPU!')
        #FIXME: problem here? why use ".module"?
        model = model.module

    validator = ImageNetVal(model, device, args)
    record = validator()

    print('Validation Finished!')
    print('Top1: ', record['top1'])
    print('Top5: ', record['top5'])
    return None