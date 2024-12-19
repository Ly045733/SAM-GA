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
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class BaseDataSets111(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'vit_H', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_testing_set = []
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        image = np.array([image, image, image]).transpose(1, 2, 0) 
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]    
            sample = {'image': image, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
class RandomGenerator111(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]


        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble}
        return sample
class BaseDataSets_SAM(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'SAM_iteration1', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 6)]
        # fold1_testing_set = []
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]
        fold1_val_set = all_cases_set

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_pesudo_label_iteration1/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        if self.split == "train":
            pesudo_label = h5f[self.pesudo_label][:]
            image_3 = np.array([image,image,image]).transpose(1,2,0)
            sample = {'image': image, 'image_sam': image_3, 'label': label, 'scribble': scribble, 'pesudo_label': pesudo_label}
            sample = self.transform(sample)
        else:
            # scribble = scribble.astype(np.uint8)
            sample = {'image': image, 'label': label, "scribble": scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample



class RandomGenerator_SAM(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]


        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble}
        return sample



class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'vit_H', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        print(self._base_dir)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]
        # fold1_testing_set = []
        # fold1_training_set = [
        #     i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_pesudo_label_iteration2/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pesudo_label = h5f[self.pesudo_label][:]
            image_3 = np.array([image, image, image]).transpose(1,2,0)
            sample = {'image': image, 'image_sam': image_3, 'label': label, 'scribble': scribble, 'pesudo_label': pesudo_label}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample

def random_rot_flip(image, label, scribble,pesudo_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    scribble = np.rot90(scribble, k)
    pesudo_label = np.rot90(pesudo_label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    scribble = np.flip(scribble, axis=axis).copy()
    pesudo_label = np.flip(pesudo_label, axis=axis).copy()
    return image, label, scribble, pesudo_label


def random_rotate(image, label, scribble, pesudo_label,  cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, 
                           reshape=False)
    label = ndimage.rotate(label, angle, order=0, 
                           reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    pesudo_label = ndimage.rotate(pesudo_label, angle, order=0, 
                           reshape=False)

    return image, label, scribble, pesudo_label


class RandomGenerator(object):
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        if self.split == 'train':
            image, image_sam, label, scribble, pesudo_label  = sample["image"], sample['image_sam'], sample["label"], sample["scribble"], sample["pesudo_label"]
            x_wotran,y_wotran = image.shape
            # ind = random.randrange(0, img.shape[0])

            # image = img[ind, ...]
            # label = lab[ind, ...]

            flag = 0
            if random.random() > 0.5:
                image, label, scribble, pesudo_label = random_rot_flip(image, label, scribble, pesudo_label)
                flag=1
            elif random.random() > 0.5:
                if 4 in np.unique(scribble):
                    image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=4)
                else:
                    image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=0)
            x, y= image.shape
            x_PL,y_PL = pesudo_label.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pesudo_label = zoom(pesudo_label, (self.output_size[0] / x_PL, self.output_size[1] / y_PL), order=0)
            image_sam = zoom(image_sam, (self.output_size[0] / x_wotran, self.output_size[1] / y_wotran, 1), order=0)


            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image_sam = torch.from_numpy(image_sam.astype(np.float32)).permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.uint8))
            pesudo_label = torch.from_numpy(pesudo_label.astype(np.uint8))
            scribble = torch.from_numpy(scribble.astype(np.uint8))

            sample = {"image": image, "label": label, "scribble":scribble,'image_sam':image_sam, 'pesudo_label': pesudo_label}
        else:
            image, label = sample['image'], sample['label']
            # image, label,scribble = sample['image'], sample['label'],sample['scribble']
            channels, x, y = image.shape
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
            # scribble = zoom(scribble, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.uint8))
            # scribble = torch.from_numpy(scribble.astype(np.uint8))
            
            # sample = {'image': image, 'label': label,'scribble':scribble}
            sample = {'image': image, 'label': label}
        return sample
    
    
class EdgeRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, edge = sample['image'], sample['label'], sample['edge']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     if 2 in np.unique(label):
        #         image, label = random_rotate(image, label, cval=2)
        #     else:
        #         image, label = random_rotate(image, label, cval=0)
        x, y  = image.shape
        zoom_factors = (self.output_size[0] / x, self.output_size[1] / y)  # 不缩放通道维度
        image = zoom(image, zoom_factors, order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 假设标签是单通道的
        edge = zoom(edge, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 将numpy数组转换为torch张量
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # 调整通道顺序为(C, H, W)
        label = torch.from_numpy(label.astype(np.uint8))
        edge = torch.from_numpy(edge.astype(np.uint8))
        # convert 255 to 1
        edge[edge == 255] = 1

        # 创建包含处理后图像和标签的字典
        sample = {'image': image, 'label': label, 'edge': edge}

        return sample


class BaseDataSets_G_SAM_PL(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'vit_H', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 21)]
        # fold1_training_set = [
        #     i for i in all_cases_set if i not in fold1_testing_set]
        fold1_testing_set = []
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_pesudo_label_iteration1/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pesudo_label = h5f[self.pesudo_label][:]
            image_3 = np.array([image, image, image]).transpose(1,2,0)
            sample = {'image': image, 'image_sam': image_3, 'label': label, 'scribble': scribble, 'pesudo_label': pesudo_label}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample


class BaseDataSets_SAM_pred(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'SAM_iteration1', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 6)]
        fold1_testing_set = []
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')

        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]  
        label = h5f['label'][:]
        scribble = h5f['scribble'][:]
        if self.split == "train":
            # pesudo_label = h5f[self.pesudo_label][:]
            image_3 = np.array([image,image,image]).transpose(1,2,0)
            sample = {'image': image, 'image_sam': image_3, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
        else:
            # scribble = scribble.astype(np.uint8)
            sample = {'image': image, 'label': label, "scribble": scribble}
            sample = self.transform(sample)
        sample["idx"] = case
        return sample
    

class RandomGenerator_SAM_pred(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        
        image,  label, scribble  = sample["image"], sample["label"], sample["scribble"]

        x, y  = image.shape
        # print(...)
        # print(...)
        # print(...)
        

        # print(image.shape)
        # print(label.shape)
        # print(scribble.shape)
        zoom_factors = (self.output_size[0] / x, self.output_size[1] / y)  # 不缩放通道维度
        image = zoom(image, zoom_factors, order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 假设标签是单通道的
        scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # print(image.shape)
        # print(label.shape)
        # print(scribble.shape)
        # print(...)
        # print(...)
        # print(...)
        
        image = torch.from_numpy(image.astype(np.float32))
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

    print(train_dataset[13]['image'].shape)
    # print(np.unique(train_dataset[1]['image'].cpu().detach().numpy()))

    print(train_dataset[13]['label'].shape)
    print(np.unique(train_dataset[1]['label']))

    print(train_dataset[13]['scribble'].shape)
    print(np.unique(train_dataset[1]['scribble']))

    print(train_dataset[13]['pesudo_label'].shape)
    print(np.unique(train_dataset[1]['pesudo_label']))
