import argparse
import os, sys
sys.path.append('./')

import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
import random, pdb, math, copy
import pickle
from utils import *
from torch import autograd

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def office_load_idx(args):
    train_bs = args.batch_size

    if args.home == True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'Art'
        elif ss == 'c':
            s = 'Clipart'
        elif ss == 'p':
            s = 'Product'
        elif ss == 'r':
            s = 'Real_World'

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'

        s_tr, s_ts = './data/office-home/{}.txt'.format(
            s), './data/office-home/{}.txt'.format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)
        tv_size = int(0.8 * dsize)
        print(dsize, tv_size, dsize - tv_size)
        s_tr, s_ts = torch.utils.data.random_split(txt_src,
                                                   [tv_size, dsize - tv_size])

        t_tr, t_ts = './data/office-home/{}.txt'.format(
            t), './data/office-home/{}.txt'.format(t)
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict['source'])
        test_source = ImageList_idx(s_ts, transform=prep_dict['source'])
        train_target = ImageList_idx(open(t_tr).readlines(),
                                 transform=prep_dict['target'])
        test_target = ImageList_idx(open(t_ts).readlines(),
                                transform=prep_dict['test'])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  #2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False)
    '''dset_loaders["source_f"] = DataLoader(fish_source,
                                           batch_size=train_bs ,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)'''
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  #3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False)
    return dset_loaders


def train_target_near(args):
    dset_loaders = office_load_idx(args)
    ## set base network

    netF = network.ResNet_sdaE().cuda()
    oldC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'
    oldC.load_state_dict(torch.load(modelpath))

    param_group_bn = []
    for k, v in netF.feature_layers.named_parameters():
        if k.find('bn') != -1:
            param_group_bn += [{'params': v, 'lr': args.lr}]
    '''{
        'params': netF.feature_layers.parameters(),
        'lr': args.lr*1
    },'''

    optimizer = optim.SGD([{
        'params': netF.bottle.parameters(),
        'lr': args.lr * 10
    }, #{  # Training or not does not matter
    #    'params': netF.em.parameters(),
    #    'lr': args.lr * 10
    #},
    {
        'params': netF.bn.parameters(),
        'lr': args.lr * 10
    }, {
        'params': oldC.parameters(),
        'lr': args.lr * 10
    }] + param_group_bn,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)

    optimizer = op_copy(optimizer)
    smax = 100

    acc_init = 0
    start = True
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output, _ = netF.forward(inputs, t=1)  
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()
            

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    oldC.train()

    while iter_num < max_iter:
        netF.train()
        oldC.train()
        iter_target = iter(dset_loaders["target"])

        try:
            inputs_test, _, indx = iter_target.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, indx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter) # learning rate decay
       
        inputs_target = inputs_test.cuda()

        output_f, masks = netF(inputs_target, t=1, s=smax)
        #print(netF.mask.max())

        masks_old = masks

        output = oldC(output_f)
        softmax_out = nn.Softmax(dim=1)(output)
        output_re = softmax_out.unsqueeze(1) 

        with torch.no_grad():
            fea_bank[indx].fill_(
                -0.1)  #do not use the current mini-batch in fea_bank
            #fea_bank=fea_bank.numpy()
            output_f_ = F.normalize(output_f).cpu().detach().clone()
            
            distance = output_f_ @ fea_bank.t()
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=2)
            score_near = score_bank[idx_near]  
            score_near = score_near.permute(0, 2, 1)

            fea_bank[indx] = output_f_.detach().clone().cpu()
            score_bank[indx] = softmax_out.detach().clone()  #.cpu()

        const = torch.log(torch.bmm(output_re, score_near)).sum(-1)
        loss_const = -torch.mean(const)

        msoftmax = softmax_out.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = im_div + loss_const 

        optimizer.zero_grad()
        loss.backward()
        # Compensate embedding gradients
        s = 100
        '''for n, p in netF.em.named_parameters():
            num = torch.cosh(torch.clamp(s * p.data, -10, 10)) + 1
            den = torch.cosh(p.data) + 1
            p.grad.data *= smax / s * num / den'''

        #print(netF.conv_final)
        for n, p in netF.bottle.named_parameters():
            if n.find('bias') == -1:
                mask_ = ((1 - masks_old)).view(-1, 1).expand(256, 2048).cuda()
                p.grad.data *= mask_
            else:  #no bias here
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

        for n, p in oldC.named_parameters():
            if args.layer == 'wn' and n.find('weight_v') != -1:
                masks__ = masks_old.view(1, -1).expand(args.class_num, 256)
                mask_ = ((1 - masks__)).cuda()
                #print(n,p.grad.shape)
                p.grad.data *= mask_
            if args.layer == 'linear':
                masks__ = masks_old.view(1, -1).expand(args.class_num, 256)
                mask_ = ((1 - masks__)).cuda()
                #print(n,p.grad.shape)
                p.grad.data *= mask_

        for n, p in netF.bn.named_parameters():
            mask_ = ((1 - masks_old)).view(-1).cuda()
            p.grad.data *= mask_

        torch.nn.utils.clip_grad_norm(netF.parameters(), 10000)

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()

            #print("target")
            acc1, _ = cal_acc_sda(dset_loaders['test'], netF, oldC, t=1)  #1
            #print("source")
            accs, _ = cal_acc_sda(dset_loaders['source_te'], netF, oldC,
                                  t=0)  # t=0
            log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%. Accuracy on source = {:.2f}%'.format(
                args.dset, iter_num, max_iter, acc1 * 100, accs * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
       

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='9',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=32,
                        help="maximum epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='c2a')
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='Office-Home')  
    parser.add_argument('--file', type=str, default='target')
    parser.add_argument('--home', action='store_true')
    parser.add_argument('--office31', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
   
    train_target_near(args)
    
