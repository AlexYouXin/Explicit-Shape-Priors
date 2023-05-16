import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk


def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(1, 2))

    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-10, 10)  # -20--20
    image_rotate_axes = [(1, 2), (2, 3), (1, 3)]
    label_rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(image, angle, axes=image_rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    label = ndimage.interpolation.rotate(label, angle, axes=label_rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)

    return image, label



def rot_from_y_x(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))  # rot along z axis, axes=(0, 1)
    label = np.rot90(label, k, axes=(0, 1))

    return image, label


def flip_xz_yz(image, label):
    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label

def intensity_shift(image):
    channel_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2), np.random.randint(2)])
    shift_value = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
    value = (channel_id * shift_value).reshape(4, 1, 1, 1)
    image = image + value
    return image
    
    
def intensity_scale(image):
    channel_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2), np.random.randint(2)])
    scale_value = np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)])
    value = (channel_id * scale_value).reshape(4, 1, 1, 1)
    image = image * value
    return image


class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.mean(np.min(image, axis=(1, 2, 3)))

        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向

        z, y, x = label.shape
        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        patch_z = np.int(self.output_size[0] / 8)
        patch_y = np.int(self.output_size[1] / 8)
        patch_x = np.int(self.output_size[2] / 8)

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        Delta_z = np.int((z_max - z_min) / 3)  # 3
        Delta_y = np.int((y_max - y_min) / 2)  # 3
        Delta_x = np.int((x_max - x_min) / 2)  # 3

        if random.random() > 0.2:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
            y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
            x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
            
        else:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
            y_random = random.randint(y_middle - Delta_y - patch_y, y_middle + Delta_y + patch_y)
            x_random = random.randint(x_middle - Delta_x - patch_x, x_middle + Delta_x + patch_x)
            
            
        
        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)


        # padding
        if crop_z_down < 0 or crop_z_up > image.shape[1]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[2]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[3]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[3]))
            image = np.pad(image, ((0, 0), (0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x
        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[:, crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

        # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image, label = rot_from_y_x(image, label)
            if random.random() > 0.5:
                image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image = intensity_shift(image)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image = intensity_scale(image)

        image = torch.from_numpy(image.astype(np.double)).float()
        label = torch.from_numpy(label.astype(np.double)).float()

        sample = {'image': image, 'label': label.long()}
        return sample


class brats_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, num_classes, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.float32)
            origin = data['origin'].astype(np.float32)
            spacing = data['space'].astype(np.float32)
            

        elif self.split == "val":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.float32)
            origin = data['origin'].astype(np.float32)
            spacing = data['space'].astype(np.float32)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.float32)
            origin = np.array(data['origin'])
            spacing = np.array(data['space'])

        label[label < 0.5] = 0.0  # maybe some voxels is a minus value
        label[label > 3.5] = 0.0
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['origin'] = origin
        sample['spacing'] = spacing
        return sample

