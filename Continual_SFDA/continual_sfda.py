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
from utils import *
import random, pdb, math, copy
import pickle
from torch import autograd




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
    if True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        t0 = 'Real_World'
        t1 = 'Art'
        t2 = 'Clipart'
        t3 = 'Product'
        dset_loaders = {}
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        for idx,t in enumerate([t0,t1,t2,t3]):
            tr, ts = './data/office-home/{}.txt'.format(
                t), './data/office-home/{}.txt'.format(t)

            txt_src = open(tr).readlines()
            dsize = len(txt_src)
            if idx==0:
                tv_size = int(0.8 * dsize)
                #print(tv_size)
                tr, ts = torch.utils.data.random_split(txt_src,
                                                    [tv_size, dsize - tv_size])
                train = ImageList_idx(tr, transform=prep_dict['source'])
                test = ImageList_idx(ts, transform=prep_dict['source'])
    
                dset_loaders[str(idx)+'tr'] = DataLoader(train,
                                                batch_size=train_bs,
                                                shuffle=True,
                                                num_workers=args.worker,
                                                drop_last=False)
                dset_loaders[str(idx)+'ts'] = DataLoader(test,
                                                batch_size=train_bs,
                                                shuffle=True,
                                                num_workers=args.worker,
                                                drop_last=False)
            else:
                tv_size = int(1 * dsize)
                tr, _ = torch.utils.data.random_split(txt_src,
                                                    [tv_size, dsize - tv_size])
                test = ImageList_idx(tr, transform=prep_dict['source'])
    
                dset_loaders[idx] = DataLoader(test,
                                                batch_size=train_bs,
                                                shuffle=True,
                                                num_workers=args.worker,
                                                drop_last=False)
            #print(dsize, tv_size, dsize - tv_size)

            
            #train = ImageList_idx(tr, transform=prep_dict['source'])
            
            print('task {} data {} loaded'.format(idx,t))
    return dset_loaders


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def train_source(args,dset_loaders):
    ## set base network
    netF = network.ResNet_sdaE_all().cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    optimizer = optim.SGD([{
        'params': netF.feature_layers.parameters(),
        'lr': args.lr
    }, {
        'params': netF.bottle.parameters(),
        'lr': args.lr * 10
    }, {
        'params': netF.em.parameters(),
        'lr': args.lr * 10
    }, {
        'params': netF.bn.parameters(),
        'lr': args.lr * 10
    }, {
        'params': netC.parameters(),
        'lr': args.lr * 10
    }],
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)

    smax = 100

    acc_init = 0
    for epoch in range(30):
        netF.train()
        netC.train()
        iter_source = iter(dset_loaders['0tr'])
        for batch_idx, (inputs_source,
                        labels_source,_) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(
            ), labels_source.cuda()

            progress_ratio = batch_idx / (len(dset_loaders) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax

            outputs, masks = netF(inputs_source, 0, s,True)
            output0 = netC(outputs[0])
            output1 = netC(outputs[1])
            output2 = netC(outputs[2])
            output3 = netC(outputs[3])
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()
            for m in masks[2]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[3]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()
            reg /= count
            loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    output0, labels_source) + CrossEntropyLabelSmooth(
                        num_classes=args.class_num, epsilon=args.smooth)(
                            output1, labels_source)+ CrossEntropyLabelSmooth(
                        num_classes=args.class_num, epsilon=args.smooth)(
                            output2, labels_source)+ CrossEntropyLabelSmooth(
                        num_classes=args.class_num, epsilon=args.smooth)(
                            output3, labels_source) + 0.1 * reg

            optimizer.zero_grad()
            loss.backward()

            # Compensate embedding gradients
            for n, p in netF.em.named_parameters():
                num = torch.cosh(torch.clamp(s * p.data, -10, 10)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den

            torch.nn.utils.clip_grad_norm(netF.parameters(), 10000)
            optimizer.step()

        netF.eval()
        netC.eval()
        acc_s_tr1, _ = cal_acc_sda(dset_loaders['0ts'], netF, netC)
        acc_s_tr2, _ = cal_acc_sda(dset_loaders['0ts'], netF, netC,t=1)
        #acc_s_te, _ = cal_acc_(dset_loaders['source_te'], netF, netB, netC)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%({:.2f}%)'.format(
            args.dset, epoch + 1, args.max_epoch, acc_s_tr1 * 100,
            acc_s_tr2 * 100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)

        if acc_s_tr1 >= acc_init:
            acc_init = acc_s_tr1
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))
    return netF,netC


def train_target_near(args,t,netF,oldC,dset_loader,mask_old=None):
    
    print('task : {}'.format(t))
    ## set base network
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
    }, {
        'params': netF.em.parameters(),
        'lr': args.lr * 10
    },
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

    
    with torch.no_grad():
        tasks=[0,1,2,3]
        tasks.remove(t)
        t0=torch.LongTensor([tasks[0]]).cuda()
        t1=torch.LongTensor([tasks[1]]).cuda()
        t2=torch.LongTensor([tasks[2]]).cuda()
        mask_0=nn.Sigmoid()(netF.em(t0) * 100)
        mask_1=nn.Sigmoid()(netF.em(t1) * 100)
        mask_2=nn.Sigmoid()(netF.em(t2) * 100)
        mask_0[mask_0<mask_1]=mask_1[mask_0<mask_1]
        mask_0[mask_0<mask_2]=mask_2[mask_0<mask_2]
        masks_old=mask_0

    acc_init = 0
    start = True
    loader = dset_loaders[t]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,512)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output, _ = netF.forward(inputs, t=t)  # a^t
            output_norm=F.normalize(output)
            outputs = oldC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()
            #all_label = torch.cat((all_label, labels.float()), 0)
        #fea_bank = fea_bank.detach().cpu().numpy()
        #score_bank = score_bank.detach()

    max_iter = args.max_epoch * len(dset_loaders[t])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.eval()
    oldC.eval()
    acc1, _ = cal_acc_sda(dset_loaders['0ts'], netF, oldC, t=0)  #1
    print(acc1)

    netF.train()
    oldC.train()
    
    while iter_num < max_iter:
    #for epoch in range(args.max_epoch):
        netF.train()
        oldC.train()
        iter_target = iter(dset_loaders[t])

        try:
            inputs_test, _, indx = iter_target.next()
        except:
            iter_test = iter(dset_loaders[t])
            inputs_test, _, indx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        #inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        '''for _, (inputs_target, _,indx) in enumerate(iter_target):
            if inputs_target.size(0) == 1:
                continue'''
        inputs_target = inputs_test.cuda()

        output_f, masks = netF(inputs_target, t=t, s=smax)
        #print(netF.mask.max())

        '''if t ==1:
            masks_old = mask_s
        else:
            masks_old=mask_old'''

        output = oldC(output_f)
        softmax_out = nn.Softmax(dim=1)(output)
        output_re = softmax_out.unsqueeze(1)  # batch x 1 x num_class

        with torch.no_grad():
            fea_bank[indx].fill_(-0.1)    #do not use the current mini-batch in fea_bank
            #fea_bank=fea_bank.numpy()
            output_f_=F.normalize(output_f).cpu().detach().clone()
            
            distance = output_f_@fea_bank.t()
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.k)
            score_near = score_bank[idx_near]    #batch x 5 x num_class
            score_near=score_near.permute(0,2,1)

            fea_bank[indx] = output_f_.detach().clone().cpu()
            score_bank[indx] = softmax_out.detach().clone()  #.cpu()


        const=torch.log(torch.bmm(output_re,score_near)).sum(-1)
        loss_const=-torch.mean(const)

      
        msoftmax = softmax_out.mean(dim=0)
        im_div= torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = im_div + loss_const 

        optimizer.zero_grad()
        loss.backward()
        # Compensate embedding gradients
        s=100
        for n, p in netF.em.named_parameters():
            num = torch.cosh(
                torch.clamp(s * p.data, -10, 10)) + 1
            den = torch.cosh(p.data) + 1
            p.grad.data *= smax / s * num / den

        #print(netF.conv_final)
        for n, p in netF.bottle.named_parameters():
            if n.find('bias') == -1:
                mask_ = ((1 - masks_old)).view(-1, 1).expand(512,
                                                                2048).cuda()
                p.grad.data *= mask_
            else:  #no bias here
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

        for n, p in oldC.named_parameters():
            if args.layer=='wn' and n.find('weight_v') != -1:
                masks__ = masks_old.view(1, -1).expand(
                        args.class_num, 512)
                mask_ = ((1 - masks__)).cuda()
                #print(n,p.grad.shape)
                p.grad.data *= mask_
            if args.layer == 'linear':
                masks__ = masks_old.view(1, -1).expand(
                        args.class_num, 512)
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
            acc1, _ = cal_acc_sda(dset_loaders['0ts'], netF, oldC, t=0)  #1
            acc2, _ = cal_acc_sda(dset_loaders[1], netF, oldC, t=1)  #1
            acc3, _ = cal_acc_sda(dset_loaders[2], netF, oldC, t=2)  #1
            acc4, _ = cal_acc_sda(dset_loaders[3], netF, oldC, t=3)  #1

            #print("source")
            log_str = 'Task: {}, Iter:{}/{}; Accuracy on 4 domains = {:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%.'.format(
                t, iter_num, max_iter, acc1 * 100,
                acc2 * 100,acc3 * 100,acc4 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
    '''if acc1 >= acc_init:
        acc_init = acc1
        best_netF = netF.state_dict()
        best_netC = oldC.state_dict()
    
        torch.save(best_netF, osp.join(args.output_dir, "F_TBD.pt"))
        torch.save(best_netC, osp.join(args.output_dir, "C_TBD.pt"))'''
    '''
    if t ==1:
        mask_old = mask_s
        mask_old[mask_old<masks]=masks[mask_old<masks]
    else:
        mask_old[mask_old<masks]=masks[mask_old<masks]
    '''

    return mask_old, netF, oldC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=40,
                        help="maximum epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--k',
                        type=int,
                        default=3,
                        help="number of neighborhoods")
    parser.add_argument('--dset', type=str, default='r2a')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--par', type=float, default=0.1)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='continual_sfda')  #trainingC_2
    parser.add_argument('--file', type=str, default='cda')
    parser.add_argument('--home', action='store_true')
    parser.add_argument('--office31', action='store_true')
    args = parser.parse_args()
    #args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    start=True

    
    current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    
    dset_loaders = office_load_idx(args)
    if args.home:
        task = ['c', 'a','p','r']
    if args.office31:
        task = ['a', 'd', 'w']
    task_s = args.dset.split('2')[0]
    task.remove(task_s)
    task_all = [task_s + '2' + i for i in task]
    for task_sameS in task_all:
        path_task = os.getcwd() + '/' + args.output + '/seed' + str(
            args.seed) + '/' + task_sameS
        if not osp.exists(path_task):
            os.mkdir(path_task)

    #if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
    if True:
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'a+')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        #train_source(args)
        netF,oldC=train_source(args,dset_loaders)
    

    for t,dset in enumerate([
            'r2a', 'r2c', 'r2p'
    ]):

        args.dset = dset
        current_folder = "./"
        args.output_dir = osp.join(current_folder, args.output,
                                'seed' + str(args.seed), args.dset)
        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        args.out_file = open(osp.join(args.output_dir, args.file + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        #train_target(args)
        #if args.file=='cluster':
        if start:
            mask_old,netF,netC=train_target_near(args,t+1,netF,oldC,dset_loaders)
            start=False
        else:
            mask_old,netF,netC=train_target_near(args,t+1,netF,netC,dset_loaders,mask_old)
        
   
