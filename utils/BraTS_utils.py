import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
import os
import torch.nn.functional as F

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    return num_params, net

class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # print(loss.item())
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0

        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])

            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]

        return loss / self.n_classes

class DiceLoss_area(nn.Module):
    def __init__(self, n_classes, weight):
        super(DiceLoss_area, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # print(loss.item())
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:

            inputs = torch.sigmoid(inputs)


        weight = self.weight
        # sigmoid

        dice_loss_0 = self._dice_loss(inputs[:, 0].float(), (target == 0).float())
        dice_loss_1 = self._dice_loss(inputs[:, 1].float(), (target == 3).float())
        dice_loss_2 = self._dice_loss(inputs[:, 2].float(), ((target == 3) | (target == 1)).float())
        dice_loss_3 = self._dice_loss(inputs[:, 3].float(), (target != 0).float())
        
        loss = dice_loss_0 * weight[0] + dice_loss_1 * weight[1] + dice_loss_2 * weight[2] + dice_loss_3 * weight[3]
        
        return loss / 4


class cross_loss_area(nn.Module):
    def __init__(self, n_classes, weight):
        super(cross_loss_area, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.ce_loss = nn.BCEWithLogitsLoss()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


    def forward(self, inputs, target):
        weight = self.weight
        # sigmoid

        cross_loss_0 = self.ce_loss(inputs[:, 0].float(), (target == 0).float())
        cross_loss_1 = self.ce_loss(inputs[:, 1].float(), (target == 3).float())
        cross_loss_2 = self.ce_loss(inputs[:, 2].float(), ((target == 3) | (target == 1)).float())
        cross_loss_3 = self.ce_loss(inputs[:, 3].float(), (target != 0).float())
        
        loss = cross_loss_0 * weight[0] + cross_loss_1 * weight[1] + cross_loss_2 * weight[2] + cross_loss_3 * weight[3]
        
        return loss / 4




def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95

    else:
        return 0, 30


def calculate_BraTS(pred, gt):
    # ET: label 3
    flag = 0   # To identify whether there are voxels with label 3
    if 3 in gt:
        flag = 1
        pred_ET = pred.copy()
        pred_ET[pred_ET < 3] = 0
        pred_ET[pred_ET == 3] = 1
        gt_ET = gt.copy()
        gt_ET[gt_ET < 3] = 0
        gt_ET[gt_ET == 3] = 1

        
        ET_dice = metric.binary.dc(pred_ET, gt_ET)
        ET_hd95 = metric.binary.hd95(pred_ET, gt_ET)
        
    else:
        ET_dice = 0
        ET_hd95 = 0
    
    # TC: label 3 and 1
    pred_TC = pred.copy()
    pred_TC[pred_TC == 3] = 1
    pred_TC[pred_TC != 1] = 0
    gt_TC = gt.copy()
    gt_TC[gt_TC == 3] = 1
    gt_TC[gt_TC != 1] = 0
    
    TC_dice = metric.binary.dc(pred_TC, gt_TC)
    TC_hd95 = metric.binary.hd95(pred_TC, gt_TC)
    
    # WT: label 3 and 1 and 2
    pred_WT = pred.copy()
    pred_WT[pred_WT > 0] = 1
    pred_WT[pred_WT < 0] = 0
    gt_WT = gt.copy()
    gt_WT[gt_WT > 0] = 1
    gt_WT[gt_WT < 0] = 0
    
    WT_dice = metric.binary.dc(pred_WT, gt_WT)
    WT_hd95 = metric.binary.hd95(pred_WT, gt_WT)

    area_dice = np.array((ET_dice, TC_dice, WT_dice))
    area_hd = np.array((ET_hd95, TC_hd95, WT_hd95))

    return area_dice, area_hd, flag



def threshold_clip(pred, threshold):
    prediction = np.zeros_like(pred)
    pred_0 = pred[0]
    pred_1 = pred[1]
    pred_2 = pred[2]
    pred_3 = pred[3]
    
    # threshold
    pred_0[pred_0 >= 0.5] = 1
    pred_0[pred_0 < 0.5] = 0
    
    # pred_1单独处理
    pred_1[pred_1 >= threshold] = 1
    pred_1[pred_1 < 0.5] = 0
    pred_2[(pred_1 >= 0.5) & (pred_1 < threshold)] = 1
    pred_3[(pred_1 >= 0.5) & (pred_1 < threshold)] = 1
    # pred_23[np.where((pred_1 >= 0.5) & (pred_1 < threshold))] = 1
    pred_1[(pred_1 >= 0.5) & (pred_1 < threshold)] = 0
    
    pred_2[pred_2 >= 0.5] = 1
    pred_2[pred_2 < 0.5] = 0
    pred_3[pred_3 >= 0.5] = 1
    pred_3[pred_3 < 0.5] = 0
    
    # concatenate
    pred_0 = np.expand_dims(pred_0, axis=0)
    pred_1 = np.expand_dims(pred_1, axis=0)
    pred_2 = np.expand_dims(pred_2, axis=0)
    pred_3 = np.expand_dims(pred_3, axis=0)
    prediction[0] = pred_0
    prediction[1] = pred_1
    prediction[2] = pred_2
    prediction[3] = pred_3
    return prediction


def mask_generation(prediction):
    mask_a = prediction[1]
    mask_b = prediction[2]
    mask_c = prediction[3]
    
    # strategy 1: 3---a and b     1---(a or b) - a     2---(b or c) - b
    class_3 = np.bitwise_and(mask_a, mask_b) * 3               # np.bitwise_and
    class_1 = np.bitwise_or(mask_a, mask_b) - mask_a
    class_2 = (np.bitwise_or(mask_b, mask_c) - mask_b) * 2
    
    '''
    # strategy 2: 3---a and c     1---(a or b) - a     2---(b or c) - b
    class_3 = np.bitwise_and(mask_a, mask_c) * 3               # np.bitwise_and
    class_1 = np.bitwise_or(mask_a, mask_b) - mask_a
    class_2 = (np.bitwise_or(mask_b, mask_c) - mask_b) * 2
    '''
    '''
    # strategy 2: 3---a     1---(a or b) - a     2---(b or c) - b
    class_3 = mask_a * 3               # np.bitwise_and
    class_1 = np.bitwise_or(mask_a, mask_b) - mask_a
    class_2 = (np.bitwise_or(mask_b, mask_c) - mask_b) * 2
    '''
    
    prediction_mask = class_1 + class_2 + class_3
    return prediction_mask

