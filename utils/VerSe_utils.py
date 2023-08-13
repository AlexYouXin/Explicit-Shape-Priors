import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time

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

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95

    else:
        return 0, 30




# slide window, with z axis overlap
# padding to the bottom and top part
# slide from top to bottom
def test_single_volume(image_, label_, net0, net1, net2, classes, patch_size, test_save_path=None, case=None, origin=None, spacing=None):
    image_, label_ = image_.squeeze(0).cpu().detach().numpy(), label_.squeeze(0).cpu().detach().numpy()
    print('previous image shape: ', image_.shape[0], image_.shape[1], image_.shape[2])
    label_[label_ < 0.5] = 0.0  # maybe some voxels is a minus value
    label_[label_ > 25.5] = 0.0
    
    label_ = np.round(label_)
    
    min_value = np.min(image_)
    # get non-zeros index
    index = np.nonzero(label_)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])
    
    
    
    image = image_[:, y_min: y_max, x_min: x_max]
    label = label_[:, y_min: y_max, x_min: x_max]
    
    padding_len = np.int(patch_size[0] / 2)
    # enlarge image field
    if (z_min - padding_len) >= 0 and (z_max + padding_len) <= image.shape[0]:
        image = image[z_min - padding_len: z_max + padding_len, :, :]
        label = label[z_min - padding_len: z_max + padding_len, :, :]
        
    else:
        t1 = z_min
        t2 = image.shape[0] - z_max
        t = np.minimum(t1, t2)
        image = image[z_min - t: z_max + t, :, :]
        label = label[z_min - t: z_max + t, :, :]
        
    print('cropped image shape: ', image.shape[0], image.shape[1], image.shape[2])
    
    
    
    # add paddding
    if image.shape[0] < patch_size[0]:
        delta_z = patch_size[0] - image.shape[0]
        image = np.pad(image, ((0, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0)
    if image.shape[1] < patch_size[1]:
        delta_y = patch_size[1] - image.shape[1]
        image = np.pad(image, ((0, 0), (0, delta_y), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (0, delta_y), (0, 0)), 'constant', constant_values=0)
    if image.shape[2] < patch_size[2]:
        delta_x = patch_size[2] - image.shape[2]
        image = np.pad(image, ((0, 0), (0, 0), (0, delta_x)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (0, 0), (0, delta_x)), 'constant', constant_values=0)


    # step of sliding windows
    step_size_z = np.int(patch_size[0] / 8)
    step_size_y = patch_size[1]
    step_size_x = patch_size[2]

    # z_padding
    z_num = np.ceil(image.shape[0] / step_size_z).astype(int)
    delta_z = np.int(z_num * step_size_z - image.shape[0])
    if delta_z % 2 == 0:
        delta_z_d = np.int(delta_z / 2)
        delta_z_u = np.int(delta_z / 2)
    else:
        delta_z_d = np.int(delta_z / 2)
        delta_z_u = np.int(delta_z / 2) + 1
    image = np.pad(image, ((delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=min_value)
    label = np.pad(label, ((delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=0.0)

    z, y, x = image.shape[0], image.shape[1], image.shape[2]
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)

        z_num = np.int((image.shape[0] - patch_size[0]) / step_size_z) + 1
        y_num = np.ceil(y / patch_size[1]).astype(int)  # 3
        x_num = np.ceil(x / patch_size[2]).astype(int)  # 4


        # construct step matrix
            
        if y_num != 1:
            step_y = y - (y_num - 1) * patch_size[1]
            
        if x_num != 1:
            step_x = x - (x_num - 1) * patch_size[2]



        torch.cuda.synchronize()
        start_time = time.time()
        net0.eval()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            pred = np.zeros((classes, image.shape[0], image.shape[1], image.shape[2]))
        ###########
            for h in range(z_num):
                for r in range(y_num):
                    for c in range(x_num):
                        # up and down

                        z_l = h * step_size_z
                        z_r = z_l + patch_size[0]
                        
                        if r == 0:
                            y_l = 0
                        elif r == y_num - 1:    
                            y_l = (r - 1) * patch_size[1] + step_y
                        else:
                            y_l = r * patch_size[1]
                            
                        y_r = y_l + patch_size[1]
                        
                        if c == 0:
                            x_l = 0
                        elif c == x_num - 1:    
                            x_l = (c - 1) * patch_size[2] + step_x
                        else:
                            x_l = c * patch_size[2]
                            
                        x_r = x_l + patch_size[2]
                        
                        slice = image[z_l: z_r, y_l: y_r, x_l: x_r]

                        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(
                            0).float().cuda()

                        outputs0 = net0(input)
                        outputs1 = net1(input)
                        outputs2 = net2(input)
                        
                        outputs0 = torch.softmax(outputs0, dim=1).squeeze(0)
                        outputs1 = torch.softmax(outputs1, dim=1).squeeze(0)
                        outputs2 = torch.softmax(outputs2, dim=1).squeeze(0)
                        
                        outputs = outputs0 + outputs1 + outputs2
                        outputs = outputs.cpu().detach().numpy()

    
                        pred[:, z_l: z_r, y_l: y_r, x_l: x_r] += outputs


            out = np.argmax(pred, axis=0)
            prediction = out[0: z, 0: y, 0: x]
    
    
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_cost = end_time - start_time
        
    
    index = np.nonzero(label)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])
    
    flatten_label = label.flatten()
    list_label = flatten_label.tolist()
    set_label = set(list_label)
    print('different values:', set_label)
    length = len(set_label)

    list_label_ = list(set_label)
    list_label_ = np.array(list_label_).astype(np.int)
    
    
    index = np.zeros(classes)
    metric_list = np.zeros((classes, 2))
    for i in range(1, length):
        metric_list[list_label_[i], :] = calculate_metric_percase(prediction[z_min: z_max, y_min: y_max, x_min: x_max] == list_label_[i], label[z_min: z_max, y_min: y_max, x_min: x_max] == list_label_[i])
        index[list_label_[i]] += 1
    

    binary_map = prediction[z_min: z_max, y_min: y_max, x_min: x_max].copy()
    binary_map[binary_map >= 1] = 1
    binary_map[binary_map < 1] = 0
    
    binary_label = label[z_min: z_max, y_min: y_max, x_min: x_max].copy()
    binary_label[binary_label >= 1] = 1
    binary_label[binary_label < 1] = 0
    
    binary_metric = calculate_metric_percase(binary_map, binary_label)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        
        
        origin = origin.flatten()
        spacing = spacing.flatten()
        origin = origin.numpy()
        spacing = spacing.numpy()
        
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
