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



def test_single_volume(image_, label_, net0, net1, net2, classes, patch_size, test_save_path=None, case=None, origin=None, spacing=None, threshold=None):   # patch_size: [256, 256]
    image_, label_ = image_.squeeze(0).cpu().detach().numpy(), label_.squeeze(0).cpu().detach().numpy()
    edge = 3
    # preprocess
    label_[label_ < 0.5] = 0.0  # maybe some voxels is a minus value
    label_[label_ > 3.5] = 0.0

    c, z, y, x = image_.shape[0], image_.shape[1], image_.shape[2], image_.shape[3]
    print('previous image shape: ', image_.shape[1], image_.shape[2], image_.shape[3])
    
    min_value = np.mean(np.min(image_, axis=(1, 2, 3)))


    
    image = image_
    label = label_
 
    
    step_size_z = np.int(patch_size[0]/4)
    step_size_y = np.int(patch_size[1]/4)
    step_size_x = np.int(patch_size[2]/4)
    
    
    if len(image.shape) == 4:


        z_num = np.ceil(image.shape[1] / step_size_z).astype(int)
        y_num = np.ceil(image.shape[2] / step_size_y).astype(int)
        x_num = np.ceil(image.shape[3] / step_size_x).astype(int)

        # add padding to size： n * step
        delta_z = np.int(z_num * step_size_z - image.shape[1])
        delta_y = np.int(y_num * step_size_y - image.shape[2])
        delta_x = np.int(x_num * step_size_x - image.shape[3])

        # z_padding
        if delta_z % 2 == 0:
            delta_z_d = np.int(delta_z / 2)
            delta_z_u = np.int(delta_z / 2)
        else:
            delta_z_d = np.int(delta_z / 2)
            delta_z_u = np.int(delta_z / 2) + 1
        image = np.pad(image, ((0, 0), (delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=0.0)
        # y_padding
        if delta_y % 2 == 0:
            delta_y_d = np.int(delta_y / 2)
            delta_y_u = np.int(delta_y / 2)
        else:
            delta_y_d = np.int(delta_y / 2)
            delta_y_u = np.int(delta_y / 2) + 1
        image = np.pad(image, ((0, 0), (0, 0), (delta_y_d, delta_y_u), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (delta_y_d, delta_y_u), (0, 0)), 'constant', constant_values=0.0)
        # x_padding
        if delta_x % 2 == 0:
            delta_x_d = np.int(delta_x / 2)
            delta_x_u = np.int(delta_x / 2)
        else:
            delta_x_d = np.int(delta_x / 2)
            delta_x_u = np.int(delta_x / 2) + 1
        image = np.pad(image, ((0, 0), (0, 0), (0, 0), (delta_x_d, delta_x_u)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (0, 0), (delta_x_d, delta_x_u)), 'constant', constant_values=0.0)
        
        print('padding image shape:', image.shape[1], image.shape[2], image.shape[3])

        prediction = np.zeros_like(label)

        z_num = np.int((image.shape[1] - patch_size[0]) / step_size_z) + 1
        y_num = np.int((image.shape[2] - patch_size[1]) / step_size_y) + 1
        x_num = np.int((image.shape[3] - patch_size[2]) / step_size_x) + 1

        ######
        torch.cuda.synchronize()
        start_time = time.time()
    
        origin = origin.flatten()
        spacing = spacing.flatten()
        origin = origin.numpy()
        spacing = spacing.numpy()

        
        with torch.no_grad():
            pred = np.zeros((classes, image.shape[1], image.shape[2], image.shape[3]))
            index = np.zeros((classes, image.shape[1], image.shape[2], image.shape[3]))
            index_list = np.zeros((4, classes, image.shape[1], image.shape[2], image.shape[3]))
            center_list = np.zeros((4, classes, image.shape[1], image.shape[2], image.shape[3]))
            ###########
            for h in range(z_num):
                for r in range(y_num):
                    for c in range(x_num):
                        # numpy to tensor
                        input = torch.from_numpy(image[:, h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]]).unsqueeze(0).float().cuda()

                        outputs0 = net0(input)  # not do slices in tensor
                        outputs1 = net1(input)
                        outputs2 = net2(input)


                        outputs0 = torch.sigmoid(outputs0).squeeze(0)
                        outputs1 = torch.sigmoid(outputs1).squeeze(0)
                        outputs2 = torch.sigmoid(outputs2).squeeze(0)
                        


                        outputs = (outputs0 + outputs1 + outputs2) / 3
                        outputs = outputs.cpu().detach().numpy()

                        index[:, h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]] += 1
                        pred[:, h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]] += outputs
                                                 


                                                 
            pred = pred / index
            pred = threshold_clip(pred, threshold)

            # out = np.argmax(pred, axis=0)
            prediction = pred
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_cost = end_time - start_time


    image = image[:, delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    prediction = prediction[:, delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    label = label[delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    index = np.nonzero(label)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])
    
    # POSTPROCESS
    
    
    # To recode label 3, label 1&3, label 1&2&3
    index = np.zeros(classes)
    metric_list = np.zeros((classes, 2))
    


    if 3 in label:
        metric_list[1, :] = calculate_metric_percase(prediction[1], label == 3)
        index[1] += 1
    if (3 in label) or (1 in label):
        metric_list[2, :] = calculate_metric_percase(prediction[2], np.bitwise_or(label == 3, label == 1))
        index[2] += 1

    metric_list[3, :] = calculate_metric_percase(prediction[3], np.bitwise_or(np.bitwise_or(label == 3, label == 1), label == 2))
    index[3] += 1

    # 4 channel mask -> three masks for label 1, 2, 3
    binary_map = prediction[3].copy()
    binary_map[binary_map >= 1] = 1
    binary_map[binary_map < 1] = 0
    
    binary_label = label.copy()
    binary_label[binary_label >= 1] = 1
    binary_label[binary_label < 1] = 0
    
    binary_metric = calculate_metric_percase(binary_map, binary_label)



    class_prediction = mask_generation(prediction.astype(np.int))
    flatten_label = class_prediction.flatten()
    list_label = flatten_label.tolist()
    set_label = set(list_label)
    print('different values:', set_label)


    class_prediction[class_prediction == 3] = 4
    label[label == 3] = 4
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image[0].astype(np.double))
        prd_itk = sitk.GetImageFromArray(class_prediction.astype(np.double))
        lab_itk = sitk.GetImageFromArray(label.astype(np.double))
        
        print('origin and spacing: ', origin, spacing)   
        
        img_itk.SetOrigin(origin)
        img_itk.SetSpacing(spacing)
        prd_itk.SetOrigin(origin)
        prd_itk.SetSpacing(spacing)
        lab_itk.SetOrigin(origin)
        lab_itk.SetSpacing(spacing)

        
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list, index, binary_metric, time_cost

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

