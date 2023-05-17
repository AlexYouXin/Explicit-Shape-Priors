import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.dataset_acdc import acdc_dataset
from utils.ACDC_utils import test_single_volume
from networks.ACDC.unet import CONFIGS as CONFIGS_ViT_seg
from networks.ACDC.unet import network as network


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/medical data/ACDC', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--edge_path', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/medical data/ACDC/edge', help='edge dir for train and val data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='/lustre/home/acct-medcb/medcb-cb1/yx/shape_prior/SPM1/ACDC/3dunet/lists/lists_Synapse', help='list dir')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=2000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,                     # 24
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=[20, 256, 256], help='input patch size of network input')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--is_savenii', action="store_true", default='True', help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='segmentation network learning rate')       # 0.01
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()




def inference(args, model0, model1, model2, model3, model4, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", num_classes=args.num_classes, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    metric_list = 0.0
    index = 0.0
    binary_metric_list = 0.0
    total_time = 0.0
    ave_dice = 0                  # 记录按case计算的均值
    ave_hd = 0
    dice_case = []                  # 记录每个case的数值
    hd_case = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 1st dimension: batch_size      2nd: z   3rd: x  4th: y
        l, h, w = sampled_batch["image"].size()[1:]
        image, label, case_name, origin, spacing = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0], sampled_batch["origin"], sampled_batch["spacing"]
        metric_i, index_i, binary_metric_i, time_i = test_single_volume(image, label, model0, model1, model2, model3, model4, classes=args.num_classes, patch_size=args.img_size,         # maybe changed into the x and y dimensions
                                      test_save_path=test_save_path, case=case_name, origin=origin, spacing=spacing)
        
        
        binary_metric_list += np.array(binary_metric_i)
        # mean for a single class, a case for all slices
        num = np.count_nonzero(index_i)
        mean_dice = np.sum(metric_i, axis=0)[0] / num
        mean_hd = np.sum(metric_i, axis=0)[1] / num
        logging.info('idx: %d, case: %s, mean_dice: %f, mean_hd95: %f' % (i_batch, case_name, mean_dice, mean_hd))
        logging.info('idx: %d, case: %s, binary dice: %f, binary hd95: %f' % (i_batch, case_name, binary_metric_i[0], binary_metric_i[1]))
        ave_dice = ave_dice + mean_dice
        ave_hd = ave_hd + mean_hd
        dice_case.append(mean_dice)
        hd_case.append(mean_hd)
        index += index_i
        metric_list += metric_i
        total_time += time_i
    # mean for all cases, but a single class
    # metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):                  # 从1开始
        metric_list[i, :] = metric_list[i, :] / index[i]
        logging.info('Mean class: %d, mean_dice %f, mean_hd95 %f' % (i, metric_list[i][0], metric_list[i][1]))
        
    binary_metric_list = binary_metric_list / len(db_test)
    mean_time = total_time / len(db_test)
    # mean for all classes and all cases
    performance = np.sum(metric_list, axis=0)[0] / (args.num_classes - 1)
    mean_hd95 = np.sum(metric_list, axis=0)[1] / (args.num_classes - 1)
    logging.info('Testing performance on classes: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    logging.info('Testing performance on classes: binary mean_dice : %f binary mean_hd95 : %f' % (binary_metric_list[0], binary_metric_list[1]))
    ave_dice = ave_dice / len(db_test)
    ave_hd = ave_hd / len(db_test)
    logging.info('Testing performance on cases: mean_dice : %f mean_hd95 : %f' % (ave_dice, ave_hd))

    logging.info('Testing time in best val model: %f' % (mean_time))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'ACDC': {
            'Dataset': acdc_dataset,
            'volume_path': '/lustre/home/acct-eeyj/eeyj-wr/youxin/medical_dataset/ACDC',
            'list_dir': '/lustre/home/acct-eeyj/eeyj-wr/youxin/shape_prior_module/LG-SPM/ACDC/3dunet/lists/lists_Synapse',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # lr
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 100000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path

    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size[0]/args.vit_patches_size), int(args.img_size[1]/args.vit_patches_size), int(args.img_size[2]/args.vit_patches_size))
    config_vit.n_patches = int(args.img_size[0] / 4) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.h = int(args.img_size[0] / 4)
    config_vit.w = int(args.img_size[1] / args.vit_patches_size)
    config_vit.l = int(args.img_size[2] / args.vit_patches_size)
    
    net0 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net1 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net2 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net3 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net4 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    
    if args.n_gpu > 1:
        net0 = nn.DataParallel(net0)
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)
        net3 = nn.DataParallel(net3)
        net4 = nn.DataParallel(net4)

    snapshot0 = os.path.join(snapshot_path, 'best_model.pth')
    snapshot1 = os.path.join(snapshot_path, 'epoch_1450.pth')  
    snapshot2 = os.path.join(snapshot_path, 'epoch_1500.pth')
    snapshot3 = os.path.join(snapshot_path, 'epoch_1400.pth')  
    snapshot4 = os.path.join(snapshot_path, 'best_model.pth')

    
    
    print(snapshot0)
    if not os.path.exists(snapshot0): snapshot0 = snapshot0.replace('best_model', 'epoch_'+str(args.max_epochs))
    print(snapshot0)
    
    net0.load_state_dict(torch.load(snapshot0))
    net1.load_state_dict(torch.load(snapshot1))
    net2.load_state_dict(torch.load(snapshot2))
    net3.load_state_dict(torch.load(snapshot3))
    net4.load_state_dict(torch.load(snapshot4))
    
    
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net0, net1, net2, net3, net4, test_save_path)


