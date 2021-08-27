import torch
import torch.nn as nn
import torchvision

import numpy as np
import os, pickle, tqdm, time, pprint

from models import get_ori_model
from svpnn import get_svpnn
from .val import ImageNetVal, accuracy

# Global parameters
save_train_epochs=.2 # how often save output during training
save_val_epochs=.5 # how often save output during validation
save_model_epochs=1 # how often save model weights
save_model_secs=720 * 10  # how often save model (in sec)
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]


class ImageNetTrain(object):

    def __init__(self, model, device, args):
        self.name = 'train'
        self.model = model
        self.args = args
        self.device = device
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == 'stepLR':
            self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.step_factor, step_size=args.step_size)
        elif args.optimizer == 'plateauLR':
            self.lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=args.step_factor, patience=args.step_size-1, threshold=0.01)
        self.loss = nn.CrossEntropyLoss()
        if args.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.args.in_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.args.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()
        if self.args.optimizer == 'stepLR':
            self.lr.step(epoch=frac_epoch)
        target = target.to(self.device)

        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        # record['learning_rate'] = self.lr.get_lr()[0]
        record['learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


def train(args = None):

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
        #TODO: finish svpnn
        model = get_svpnn()
    
    if args.ngpus > 0 :
        print('We have {} GPU(s) detected'.format(str(torch.cuda.device_count())))
        model = model.to(device)
    else:
        print('Caution!!! We run on CPU!')
        #FIXME: problem here? why use ".module"?
        model = model.module

    trainer = ImageNetTrain(model, device, args)
    validator = ImageNetVal(model, device, args)

    # start training
    start_epoch = 0
    records = []

    ## load model
    if args.restore_epoch > 0 and args.restore_path:
        print('Restoring epoch {e} from {p}'.format(e=str(args.restore_epoch), p=args.restore_path))
        weights_path = os.path.join(args.restore_path, '{m}_epoch_{e:02d}.pth.tar'.format(m=args.model_arch, e=args.restore_epoch))

        ckpt_data = torch.load(weights_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
        results_old = pickle.load(open(os.path.join(args.restore_path, 'results.pkl'), 'rb'))
        for result in results_old:
            records.append(result)
    
    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    save_train_steps = (np.arange(0, args.epochs+1, save_train_epochs)*nsteps).astype(int)
    save_val_steps = (np.arange(0, args.epochs+1, save_val_epochs)*nsteps).astype(int)
    save_model_steps = (np.arange(0, args.epochs+1, save_model_epochs)*nsteps).astype(int)

    ## iteration
    for epoch in tqdm.trange(start_epoch, arge.epochs+1, initial=0, desc='epoch'):
        print('----- Epoch: {} -----'.format(epoch))
        data_load_start = np.nan
        data_loader_iter = trainer.data_loader

        for step, data in enumerate(tqdm.tqdm(data_loader_iter, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * nsteps + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    if args.optimizer == 'plateauLR' and step == 0:
                        trainer.lr.step(results[validator.name]['loss'])
                    trainer.model.train()
                    print('LR: ', trainer.optimizer.param_groups[0]["lr"])

            if args.output_path is not None:
                if not (os.path.isdir(args.output_path)):
                    os.mkdir(args.output_path)

                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(args.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                # ckpt_data['args'] = args.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(args.output_path,'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(args.output_path,'{m}_epoch_{e:02d}.pth.tar'.format(m=args.model_arch, e=epoch)))

            elif len(results) > 1:
                pprint.pprint(results)

            if epoch < args.epochs:
                frac_epoch = (global_step + 1) / nsteps
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()

    print('Training Finished!')
    return None