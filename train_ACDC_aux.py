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

from dataset.dataset_ACDC import BaseDataSets, RandomGenerator
from Networks.net_factory import net_factory
# from utils import losses, metrics, ramps
# from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from my_utils.val2D import test_single_volume_cct, test_single_volume_ds,test_single_volume_cct_tree
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/cj/code/SAM_Scribble/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test/ACDC/noCE_InterFA_SA_Two_thin_aux', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold5', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct_SA_thin_aux', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


def train(args, snapshot_path, savepath):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size,split='train')
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold,transform=transforms.Compose([
        RandomGenerator(args.patch_size,split='val')
    ]), split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = pDLoss(num_classes, ignore_index=4)
    # abl_loss = ABL(max_N_ratio = 1/50)
    # bce_loss = CrossEntropyLoss() 


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    cont1,cont2,contmix = 0,0,0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, pesudo_label_batch = sampled_batch['image'], sampled_batch['scribble'], sampled_batch['pesudo_label']

            volume_batch, label_batch, pesudo_label_batch = volume_batch.cuda(), label_batch.cuda(), pesudo_label_batch.cuda()

            outputs, outputs_aux1,min_out = model( volume_batch)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            min_out_soft = torch.softmax(min_out, dim=1)
            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            loss_cemix = ce_loss(min_out_soft, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2 + loss_cemix)


            # beta = random.random() + 1e-10
            # pseudo_supervision = torch.argmax(
            #     (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)


            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pesudo_label_batch.unsqueeze(
                1)) + dice_loss(outputs_soft2, pesudo_label_batch.unsqueeze(1))+dice_loss(min_out_soft, pesudo_label_batch.unsqueeze(
                1)))

            loss = loss_ce + 0.5 * loss_pse_sup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, alpha: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), alpha))

            if iter_num % 200 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                
                metric_list_1,metric_list_2,metric_list_mix = 0.0,0.0,0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i_1,metric_i_2,metric_i_mix = test_single_volume_cct_tree(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)
                    metric_list_mix += np.array(metric_i_mix)

                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)
                metric_list_mix = metric_list_mix / len(db_val)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list_1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list_1[class_i, 1], iter_num)
                    

                performance_1 = np.mean(metric_list_1, axis=0)[0]
                mean_hd95_1 = np.mean(metric_list_1, axis=0)[1]

                performance_2 = np.mean(metric_list_2, axis=0)[0]
                mean_hd95_2 = np.mean(metric_list_2, axis=0)[1]
                
                performance_mix = np.mean(metric_list_mix, axis=0)[0]
                mean_hd95_mix = np.mean(metric_list_mix, axis=0)[1]

                performance = max(performance_1,performance_2,performance_mix)
                if performance==performance_1:
                    cont1+=1
                    mean_hd95 = mean_hd95_1
                    dice_cls0,dice_cls1,dice_cls2 = metric_list_1[0,0],metric_list_1[1,0],metric_list_1[2,0]
                    HD95_cls0,HD95_cls1,HD95_cls2 = metric_list_1[0,1],metric_list_1[1,1],metric_list_1[2,1]
                    writer.add_scalar('info/val_mean_dice', performance, iter_num)
                    writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                elif performance==performance_2:
                    cont2+=1
                    mean_hd95 = mean_hd95_2
                    dice_cls0,dice_cls1,dice_cls2 = metric_list_2[0,0],metric_list_2[1,0],metric_list_2[2,0]
                    HD95_cls0,HD95_cls1,HD95_cls2 = metric_list_2[0,1],metric_list_2[1,1],metric_list_2[2,1]
                    writer.add_scalar('info/val_mean_dice', performance, iter_num)
                    writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                else:
                    contmix+=1
                    mean_hd95 = mean_hd95_mix
                    dice_cls0,dice_cls1,dice_cls2 = metric_list_mix[0,0],metric_list_mix[1,0],metric_list_mix[2,0]
                    HD95_cls0,HD95_cls1,HD95_cls2 = metric_list_mix[0,1],metric_list_mix[1,1],metric_list_mix[2,1]
                    writer.add_scalar('info/val_mean_dice', performance, iter_num)
                    writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)


                if performance > best_performance:
                    best_performance,best_hd95 = performance,mean_hd95
                    best_dice_0,best_dice_1,best_dice_2 = dice_cls0,dice_cls1,dice_cls2
                    best_hd95_0,best_hd95_1,best_hd95_2 = HD95_cls0,HD95_cls1,HD95_cls2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                logging.info('iteration %d : LV_dice : %f LV_hd95 : %f' % (iter_num, dice_cls0, HD95_cls0))
                logging.info('iteration %d : MYO_dice : %f MYO_hd95 : %f' % (iter_num, dice_cls1, HD95_cls1))
                logging.info('iteration %d : RV_dice : %f RV_hd95 : %f' % (iter_num, dice_cls2, HD95_cls2))
                logging.info('cont1 %d : cont2 : %f contmix : %f' % (cont1, cont2, contmix))
                
                    
                
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    import csv
    with open(os.path.join(savepath, 'best_metrics.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['','mean_dice','mean_hd95', 'cls0_dice','cls0_hd95', 'cls1_dice','cls1_hd95', 'cls2_dice','cls2_hd95'])
        writer.writerow([args.fold, best_performance,best_hd95,best_dice_0,best_hd95_0,best_dice_1,best_hd95_1,best_dice_2,best_hd95_2])   
    return "Training Finished!"


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

    savepath =  "../model_New/{}".format(
        args.exp)
    snapshot_path = "../model_New/{}/{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path,savepath)
