# coding=utf-8
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from networks.BraTS.unet import CONFIGS as CONFIGS_ViT_seg
from networks.BraTS.unet import network as network
from BraTS_trainer import run_main

parser = argparse.ArgumentParser()
parser.add_argument('--train_root_path', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/medical data/BraTS 2020/train', help='root dir for train data')
parser.add_argument('--val_root_path', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/medical data/BraTS 2020/train', help='root dir for val data')
parser.add_argument('--dataset', type=str,
                    default='BraTS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/BraTS2020/baseline/3dunet/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')        # 25 + 1
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=2000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')             # 24
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,        # 0.01
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=[128, 128, 128], help='input patch size of network input')                 # network input image size : 128
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True            # False
    dataset_name = args.dataset
    dataset_config = {
        'BraTS': {
            'train_root_path': '/lustre/home/acct-eeyj/eeyj-wr/youxin/medical_dataset/BraTS2020/train',
            'val_root_path': '/lustre/home/acct-eeyj/eeyj-wr/youxin/medical_dataset/BraTS2020/train',
            'list_dir': '/lustre/home/acct-eeyj/eeyj-wr/youxin/shape_prior_module/LG-SPM/BraTS2020/3dunet/lists/lists_Synapse',
            'num_classes': 4,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.train_root_path = dataset_config[dataset_name]['train_root_path']
    args.val_root_path = dataset_config[dataset_name]['val_root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.batch_size = args.batch_size
    # number of patches
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size), int(args.img_size[2] / args.vit_patches_size))
    ###
    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.h = int(args.img_size[0] / args.vit_patches_size)
    config_vit.w = int(args.img_size[1] / args.vit_patches_size)
    config_vit.l = int(args.img_size[2] / args.vit_patches_size)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'         # '0,1,2,3'
    torch.cuda.empty_cache()
    net = network(in_channel=4, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    
    if args.n_gpu > 1:
        net = nn.DataParallel(net)

    # train the network
    run_model = {'BraTS': run_main,}
    run_model[dataset_name](args, net, snapshot_path)

