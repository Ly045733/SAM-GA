import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# from losses.abl_loss import ABL
from losses.pd_loss import pDLoss
from matplotlib import pyplot as plt

from dataset.dataset_ACDC import BaseDataSets_G_SAM_PL, RandomGenerator
from Networks.net_factory import net_factory
from my_utils.val2D import test_single_volume_cct, test_single_volume_ds
from PIL import Image 
import imageio

root_path='/home/cj/code/SAM_Scribble/data/ACDC'
savepath = "/home/cj/code/model_New/test/ACDC/InterFA_SA_Two_thin_aux/fold4"
net_type = 'unet_cct_SA_thin_aux'

modelpath = os.path.join(savepath, "scribble/"+net_type+"_best_model.pth")
# savepath = '/home/cj/code/SAM_Scribble/pesudo_label/for_SAM/SAM_iteration111'
savepath = os.path.join(savepath, "pesudolabel/")
os.makedirs(savepath,exist_ok=True)
fold = 'fold1'
patch_size = [256,256]

num_classes = 4

model = net_factory(net_type=net_type, in_chns=1, class_num=num_classes)
# 补足模型路径
model.load_state_dict(torch.load(modelpath))
db_val = BaseDataSets_G_SAM_PL(base_dir=root_path, fold=fold,transform=transforms.Compose([
    RandomGenerator(patch_size,split='train')
]), split="train")
valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                        num_workers=0)
model.eval() 
for i_batch, sampled_batch in enumerate(tqdm(valloader)):
    image, id = sampled_batch['image'].cuda(), sampled_batch['idx'][0]
    label = sampled_batch['label'].numpy()
    id = os.path.splitext(id)[0]
    output_main,output_aux = model(image)
    out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
    out = out.detach().cpu().numpy().astype(np.uint8())



    output_image =  Image.fromarray(out, mode='L')
    output_image.save(savepath+f"/{id}.png")



    


# folder = '/home/cj/code/SAM_Scribble/pesudo_label/for_SAM/SAM_iteration111'
# for file_name in os.listdir(folder):
#     file_path = os.path.join(folder, file_name)
#     image = Image.open(file_path, mode= 'r')
#     image = image.convert('L')
#     image = np.array(image)
#     print(image.shape)
#     print(np.unique(image))






    
    

