import cv2
import h5py
import numpy as np
import torch
import os
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def pseudo_label_generator_cell(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    def rgb_to_gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    if 1 not in np.unique(seed) or 2 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 3] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        
        gray_data = rgb_to_gray(data)  # 将RGB图像转换为灰度图像
        sigma = 0.35
        gray_data = rescale_intensity(gray_data, in_range=(-sigma, 1 + sigma),
                                      out_range=(-1, 1))
        segmentation = random_walker(gray_data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label", edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list = test_ids
        
        print("total {} samples".format(len(self.sample_list)))
        
    
    def _get_fold_ids(self, fold):
        
        all_cases_set = list(range(1, 2632))
        fold1_testing_set = [i for i in range(1, 527)]
        fold1_training_set = [i for i in all_cases_set if i not in fold1_testing_set]
        
        fold2_testing_set = [i for i in range(527, 1053)]
        fold2_training_set = [i for i in all_cases_set if i not in fold2_testing_set]
        
        fold3_testing_set = [i for i in range(1053, 1579)]
        fold3_training_set = [i for i in all_cases_set if i not in fold3_testing_set]
        
        fold4_testing_set = [i for i in range(1579, 2105)]
        fold4_training_set = [i for i in all_cases_set if i not in fold4_testing_set]
        
        fold5_testing_set = [i for i in range(2105, 2632)]
        fold5_training_set = [i for i in all_cases_set if i not in fold5_testing_set]
        
        
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
        h5f = h5py.File(os.path.join(self._base_dir, str(case) + '.h5'), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_cell(image, h5f["scribble"][:])
            else:
                scribble = h5f['scribble'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label, 'scribble': scribble}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
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
    return image, label, scribble,pesudo_label


def random_rotate(image, label, scribble,pesudo_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, 
                           reshape=False)
    label = ndimage.rotate(label, angle, order=0, 
                           reshape=False)
    pesudo_label = ndimage.rotate(pesudo_label, angle, order=0, 
                           reshape=False)
    scribble = ndimage.rotate(scribble, angle, order=0,
                           reshape=False, mode="constant", cval=cval)

    return image, label, scribble,pesudo_label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, scribble = sample["image"], sample["label"], sample["scribble"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]

        # if random.random() > 0.5:
        #     image, label, scribble = random_rot_flip(image, label, scribble)
        # elif random.random() > 0.5:
        #     if 3 in np.unique(scribble):
        #         image, label, scribble = random_rotate(image, label, scribble, cval=3)
        #     else:
        #         image, label, scribble = random_rotate(image, label, scribble, cval=0)
        
        _, x, y = image.shape

        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)


        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))
        sample = {"image": image, "label": label, "scribble":scribble}
        return sample
    

class BaseDataSets_cell(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="scribble",pesudo_label = 'Vit_H', edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        self.pesudo_label = pesudo_label
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list = test_ids
        
        print("total {} samples".format(len(self.sample_list)))
        

    def _get_fold_ids(self, fold):
        all_cases_set = list(range(1, 2632))
        fold1_testing_set = [i for i in range(1, 527)]
        fold1_training_set = [i for i in all_cases_set if i not in fold1_testing_set]
        
        fold2_testing_set = [i for i in range(527, 1053)]
        fold2_training_set = [i for i in all_cases_set if i not in fold2_testing_set]
        
        fold3_testing_set = [i for i in range(1053, 1579)]
        fold3_training_set = [i for i in all_cases_set if i not in fold3_testing_set]
        
        fold4_testing_set = [i for i in range(1579, 2105)]
        fold4_training_set = [i for i in all_cases_set if i not in fold4_testing_set]
        
        fold5_testing_set = [i for i in range(2105, 2632)]
        fold5_training_set = [i for i in all_cases_set if i not in fold5_testing_set]
        
        
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

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, str(case) + '.h5'), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pl = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label':label ,'scribble': scribble, 'PL': pl}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class RandomGenerator_cell(object):
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        if self.split == 'train':
            image, scribble, label, pesudo_label  = sample["image"], sample["scribble"],sample['label'], sample["PL"]
            # ind = random.randrange(0, img.shape[0])

            # image = img[ind, ...]
            # label = lab[ind, ...]

            # flag = 0
            # if random.random() > 0.5:
            #     image, label, scribble, pesudo_label = random_rot_flip(image, label, scribble, pesudo_label)
            #     flag=1
            # elif random.random() > 0.5:
            #     if 3 in np.unique(scribble):
            #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=3)
            #     else:
            #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=0)
            # # print(image.shape)
            x, y, _= image.shape
            # x_PL,y_PL = pesudo_label.shape
            zoom_factors = (self.output_size[0] / x, self.output_size[1] / y, 1)
            image = zoom(image, zoom_factors, order=0)
            # print(image.shape)
            scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # print(scribble.shape)
            pesudo_label = zoom(pesudo_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # print(pesudo_label.shape)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # print(label.shape)


            image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
            label = torch.from_numpy(label.astype(np.uint8))
            pesudo_label = torch.from_numpy(pesudo_label.astype(np.uint8))
            scribble = torch.from_numpy(scribble.astype(np.uint8))

            sample = {"image": image, "label": label, "scribble":scribble, 'pesudo_label': pesudo_label}
        else:
            image, label = sample['image'], sample['label']
            x, y, _= image.shape
            zoom_factors = (self.output_size[0] / x, self.output_size[1] / y, 1)
            image = zoom(image, zoom_factors, order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
            label = torch.from_numpy(label.astype(np.uint8))

            sample = {'image': image, 'label': label}
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
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list = test_ids
        
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 21)]
        # fold1_training_set = [
        #     i for i in all_cases_set if i not in fold1_testing_set]
        all_cases_set = list(range(1, 2632))
        # fold1_testing_set = [i for i in range(1, 527)]
        fold1_training_set = []
        fold1_testing_set = [i for i in all_cases_set if i not in fold1_training_set]
        
        fold2_testing_set = [i for i in range(527, 1053)]
        fold2_training_set = [i for i in all_cases_set if i not in fold2_testing_set]
        
        fold3_testing_set = [i for i in range(1053, 1579)]
        fold3_training_set = [i for i in all_cases_set if i not in fold3_testing_set]
        
        fold4_testing_set = [i for i in range(1579, 2105)]
        fold4_training_set = [i for i in all_cases_set if i not in fold4_testing_set]
        
        fold5_testing_set = [i for i in range(2105, 2632)]
        fold5_training_set = [i for i in all_cases_set if i not in fold5_testing_set]
        
        
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
        # print(case)

        h5f = h5py.File(os.path.join(self._base_dir, str(case) + '.h5'), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            # pl = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label':label ,'scribble': scribble}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        sample["idx"] = case
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
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list = test_ids
        
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold1_testing_set = [
        #     "patient{:0>3}".format(i) for i in range(1, 21)]
        # fold1_training_set = [
        #     i for i in all_cases_set if i not in fold1_testing_set]
        all_cases_set = list(range(1, 2632))
        fold1_testing_set = [i for i in range(1, 200)]
        # fold1_training_set = []
        fold1_training_set = [i for i in all_cases_set if i not in fold1_testing_set]
        
        fold2_testing_set = [i for i in range(527, 1053)]
        fold2_training_set = [i for i in all_cases_set if i not in fold2_testing_set]
        
        fold3_testing_set = [i for i in range(1053, 1579)]
        fold3_training_set = [i for i in all_cases_set if i not in fold3_testing_set]
        
        fold4_testing_set = [i for i in range(1579, 2105)]
        fold4_training_set = [i for i in all_cases_set if i not in fold4_testing_set]
        
        fold5_testing_set = [i for i in range(2105, 2632)]
        fold5_training_set = [i for i in all_cases_set if i not in fold5_testing_set]
        
        
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
        # print(case)

        h5f = h5py.File(os.path.join(self._base_dir, str(case) + '.h5'), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.split == "train":
            scribble = h5f['scribble'][:]
            pl = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label':label ,'scribble': scribble,'PL':pl}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            # sample = {'image': image, 'label': label}

            scribble = h5f['scribble'][:]
            pl = h5f[self.pesudo_label][:]
            sample = {'image': image, 'label':label ,'scribble': scribble,'PL':pl}

            
            sample = self.transform(sample)
        sample["idx"] = case
        return sample

class RandomGenerator_SAM_cell(object):
    def __init__(self, output_size,split):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        
        image, scribble, label, pesudo_label  = sample["image"], sample["scribble"],sample['label'], sample["PL"]
        # ind = random.randrange(0, img.shape[0])

        # image = img[ind, ...]
        # label = lab[ind, ...]

        # flag = 0
        # if random.random() > 0.5:
        #     image, label, scribble, pesudo_label = random_rot_flip(image, label, scribble, pesudo_label)
        #     flag=1
        # elif random.random() > 0.5:
        #     if 3 in np.unique(scribble):
        #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=3)
        #     else:
        #         image, label, scribble, pesudo_label = random_rotate(image, label, scribble, pesudo_label, cval=0)
        # # print(image.shape)
        x, y, _= image.shape
        x_PL,y_PL = pesudo_label.shape
        zoom_factors = (self.output_size[0] / x, self.output_size[1] / y, 1)
        image = zoom(image, zoom_factors, order=0)
        # print(image.shape)
        scribble = zoom(scribble, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # print(scribble.shape)
        pesudo_label = zoom(pesudo_label, (self.output_size[0] / x_PL, self.output_size[1] / y_PL), order=0)
        # print(pesudo_label.shape)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # print(label.shape)


        image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)
        label = torch.from_numpy(label.astype(np.uint8))
        pesudo_label = torch.from_numpy(pesudo_label.astype(np.uint8))
        scribble = torch.from_numpy(scribble.astype(np.uint8))

        sample = {"image": image, "label": label, "scribble":scribble, 'pesudo_label': pesudo_label}
        
        return sample


if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    train_dataset = BaseDataSets_G_SAM_PL(base_dir="/home/cj/code/SAM_Scribble/data/SegPC2021/training_iteration1", split="val", transform=transforms.Compose([
        RandomGenerator_SAM_cell([256, 256],'val')]
        ), fold='fold1', sup_type='scribble',pesudo_label='vit_H')


    print(train_dataset[1]['image'].shape)
    print(np.unique(train_dataset[1]['image'].cpu().detach().numpy()))
    
    print(train_dataset[1]['label'].shape)
    print(np.unique(train_dataset[1]['label'].cpu().detach().numpy()))

    print(train_dataset[1]['scribble'].shape)
    print(np.unique(train_dataset[1]['scribble'].cpu().detach().numpy()))
    print(train_dataset[1]['pesudo_label'].shape)
    print(np.unique(train_dataset[1]['scribble'].cpu().detach().numpy()))