import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
import os

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
        # print('outputs, targets after one-hot:', inputs.shape, target.shape)           # ([1, 2, 64, 64, 64])  ([1, 2, 64, 64, 64])
        # allocate weight for segmentation of each class
        # if weight is None:
            # weight = [1] * self.n_classes
        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        # print('num classes:', self.n_classes)         # 2
        for i in range(0, self.n_classes):
            # print(inputs.shape, target.shape)              # torch.Size([1, 26, 128, 128, 128]) torch.Size([1, 26, 128, 128, 128])
            # print('min and max: ', torch.min(inputs[:, i, :, :, :]), torch.min(target[:, i, :, :, :]), torch.max(inputs[:, i, :, :, :]), torch.max(target[:, i, :, :, :]))
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            # print(dice_loss.item())
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        # return loss / self.n_classes
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95

    else:
        return 0, 30




