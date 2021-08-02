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
import shutil



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_source(args):
    dset_loaders = office_load(args)
    ## set base network
    netF = network.ResNet_sdaE().cuda()
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
    for epoch in range(args.max_epoch):
        netF.train()
        netC.train()
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source,
                        labels_source) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(
            ), labels_source.cuda()

            progress_ratio = batch_idx / (len(dset_loaders) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax

            outputs, masks = netF(inputs_source, 0, s, True)
            output0 = netC(outputs[0])
            output1 = netC(outputs[1])
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()
            reg /= count
            loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    output0, labels_source) + CrossEntropyLabelSmooth(
                        num_classes=args.class_num, epsilon=args.smooth)(
                            output1, labels_source) + 0.15 * reg

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
        acc_s_tr1, _ = cal_acc_sda(dset_loaders['source_te'], netF, netC)
        acc_s_tr2, _ = cal_acc_sda(dset_loaders['source_te'], netF, netC, t=1)
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


def test_target(args):
    dset_loaders = office_load(args)
    ## set base network
    netF = network.ResNet_sdaE().cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()

    acc, _ = cal_acc_sda(dset_loaders['test'], netF, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)


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
                        default=20,
                        help="maximum epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='a2c')
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
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.home:
        task = ['c', 'a', 'p', 'r']
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

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    file_f = osp.join(args.output_dir + '/source_F.pt')
    file_c = osp.join(args.output_dir + '/source_C.pt')
    task.remove(args.dset.split('2')[1])
    task_remain = [task_s + '2' + i for i in task]
    for task_sameS in task_remain:
        path_task = os.getcwd() + '/' + args.output + '/seed' + str(
            args.seed) + '/' + task_sameS
        pathF_copy = osp.join(path_task, 'source_F.pt')
        pathC_copy = osp.join(path_task, 'source_C.pt')
        if not osp.exists(pathF_copy):
            shutil.copy(file_f, pathF_copy)
            shutil.copy(file_c, pathC_copy)