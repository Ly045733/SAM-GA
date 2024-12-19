import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from random import sample

import pandas as pd

def pseudo_label_generator_prostate(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label =  segmentation - 1
    return pseudo_label

class MSCMRDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",
                 train_dir="/MSCMR_training_slices", val_dir="/MSCMR_training_volumes"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        train_ids, test_ids = self._get_fold_ids(fold)
        

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["patient{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        validation_set = ["patient{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_prostate(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'gt': h5f['label'][:]}
            if self.transform:
                sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.int8)}
        sample["idx"] = case
        return sample
    

def random_rot_flip(image, label, scribble):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    return image, label, scribble


def random_rotate(image, label, scribble, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, 
                           reshape=False)
    label = ndimage.rotate(label, angle, order=0, 
                           reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0,
                           reshape=False, mode="constant", cval=cval)

    return image, label, scribble


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, scribble = sample["image"], sample["label"], sample["scribble"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label, scribble = random_rot_flip(image, label, scribble)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label, scribble = random_rotate(image, label, scribble, cval=4)
            else:
                image, label, scribble = random_rotate(image, label, scribble, cval=0)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))
        sample = {"image": image, "label": label, "scribble":scribble}
        return sample


class MSCMR_BaseDataSets_SAM_pred(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'SAM_iteration1', edge_paras=None):
        def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",train_dir="/MSCMR_training_slices", val_dir="/MSCMR_training_volumes"):
        
            self._base_dir = base_dir
            self.sample_list = []
            self.split = split
            self.sup_type = sup_type
            self.transform = transform
            self.train_dir = train_dir
            self.val_dir = val_dir
            train_ids, test_ids = self._get_fold_ids(fold)
        

            if self.split == 'train':
                self.all_slices = os.listdir(
                    self._base_dir + self.train_dir)
                self.sample_list = []
                for ids in train_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)

            elif self.split == 'val':
                self.all_volumes = os.listdir(
                    self._base_dir + self.val_dir)
                self.sample_list = []
                print("test_ids", test_ids)
                for ids in test_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_volumes))
                    self.sample_list.extend(new_data_list)

            print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["patient{:0>2}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        validation_set = ["patient{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        image = h5f['image'][:]  
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        if self.split == "train":
            image_3 = np.array([image,image,image]).transpose(1,2,0)
            sample = {'image': image, 'image_sam': image_3, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label, "scribble": scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    

class MSCMR_RandomGenerator_SAM_pred(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]


        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble}
        return sample

if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    train_dataset = BaseDataSets_SAM_pred(base_dir="/home/cj/code/SAM_Scribble/data/ACDC", split="train", transform=transforms.Compose([
        BaseDataSets_SAM_pred([256, 256],'train')]
        ), fold='fold1', sup_type='scribble', edge_paras="30_40_0",pesudo_label='vit_H')
    
    print(train_dataset[1]['image'].shape)
    print(np.unique(train_dataset[1]['image'].cpu().detach().numpy()))
    
    print(train_dataset[1]['label'].shape)
    print(np.unique(train_dataset[1]['label'].cpu().detach().numpy()))

    print(train_dataset[1]['scribble'].shape)
    print(np.unique(train_dataset[1]['scribble'].cpu().detach().numpy()))