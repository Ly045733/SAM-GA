import torch
from segment_anything.modeling.sam import Sam
from segment_anything import sam_model_registry, SamPredictor
from dataset.dataset_ACDC import BaseDataSets_SAM_pred,RandomGenerator_SAM_pred
from dataset.dataset_MSCMR import MSCMR_BaseDataSets_SAM_pred,  MSCMR_RandomGenerator_SAM_pred
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

import pandas as pd
from my_utils.Sampling_Combine import random_sample, contour_sample, combine, Entropy_Grids_Sampling,Entropy_contour_Sampling
from my_utils.metrics import calculate_metrics
from matplotlib.colors import ListedColormap
from my_utils import save_weight
if __name__ == '__main__':
     
    # checkpoint = "model_path/sam_vit_h_4b8939.pth"
    # model_type =os.path.split(checkpoint)[1][4:9]
    # save_best_path = '/home/cj/code/model_New/SAM_FineTune/iteration1/sam_best_decoder.pth'
    # sam_model = sam_model_registry[model_type](checkpoint).cuda()
    # sam_model = save_weight.load_partial_weight(
    #         model=sam_model,
    #         load_path=save_best_path
    #     )
    # predictor = SamPredictor(sam_model)
    dataset = BaseDataSets_SAM_pred(base_dir="/home/cj/code/SAM_Scribble/data/ACDC", transform=transforms.Compose([
                                RandomGenerator_SAM_pred([256, 256])]
                                ),split="train",  fold='fold1', sup_type='scribble')
    dataloader = DataLoader(dataset, 1 , shuffle= False)

    experiment_name = "MSCMR_Contour_Sampling_P5_N5_small_VitH_iteration1"
    result_path = f"result/{experiment_name}" + "/pesudo_label_image"

    os.makedirs(result_path, exist_ok = True)


    num = 5
    num_class = 4   
    num_batches = len(dataloader)

    total_dice_scores, total_HD95_scores, total_IOU_scores = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    for step, sample_list in tqdm(enumerate(dataloader)):

        image, label, scribble, id  = sample_list['image'].cuda(), sample_list['label'].cuda(), sample_list['scribble'].cuda(), sample_list['idx']
        
        # sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        # process_img = sam_transform.apply_image_torch(image)
        
        # sampled_point_batch_LV =  Entropy_contour_Sampling(scribble, image, 1, num)
        # sampled_point_batch_MYO = Entropy_contour_Sampling(scribble, image, 2, num)
        # sampled_point_batch_RV =  Entropy_contour_Sampling(scribble, image, 3, num)
        # sampled_point_batch_background = Entropy_contour_Sampling(scribble, image, 4, num)

        # sampled_point_batch_LV =  contour_sample(scribble, 1, num)
        # sampled_point_batch_MYO = contour_sample(scribble, 2, num)
        # sampled_point_batch_RV =  contour_sample(scribble, 3, num)
        # sampled_point_batch_background = contour_sample(scribble, 4, num)
        # print(image.shape)
        image = (np.array(image[0].cpu())*255).astype(np.uint8)
        # label = label[0].cpu().numpy()
    #     scribble = scribble[0].cpu().numpy()
    #     color  = [(1,1,0,1),
    #               (1,0,0,1),
    #               (0,1,0,1),
    #               (0,0,1,1),
    #               (0,0,0,0),
    #               ]
    #     cmap = ListedColormap(color)
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image,cmap='gray')
    #     plt.axis('off')
    # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #     plt.imshow(scribble,cmap=cmap,alpha=0.5)
    #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
    #     plt.savefig(f'data/444/{id[0]}_mask_.png', bbox_inches='tight', pad_inches=0) 
    #     plt.close()

        label1 = label[0].cpu().numpy()
        label1[label1 == 3] = 255
        label1[label1 == 2] = 128
        label1[label1 == 1] = 64
        


        im = Image.fromarray(label1)
        im.convert('L').save(f'/home/cj/code/SAM_Scribble/vis/ACDC/img_label/label/{id[0][:-3]}.png')

        im = Image.fromarray(image)
        im.convert('L').save(f'/home/cj/code/SAM_Scribble/vis/ACDC/img_label/image/{id[0][:-3]}.png')

        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'data/image/{id[0]}_label_RV.png', bbox_inches='tight', pad_inches=0)


        # plt.imshow(label1, cmap='gray', alpha= 0.5)

        # plt.axis('off')
        # plt.savefig(f'data/111/{id[0]}_label_RV.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO = combine(sampled_point_batch_LV, sampled_point_batch_RV, sampled_point_batch_MYO, sampled_point_batch_background)
        
        # if all_points_LV is not None:
        #     input_points_LV,  input_labels_LV  = np.array(all_points_LV),  np.array(all_labels_LV)
        # else: 
        #     input_points_LV = None
        #     input_labels_LV = None
        # if all_points_RV is not None:
        #     input_points_RV,  input_labels_RV  = np.array(all_points_RV),  np.array(all_labels_RV)
        # else: 
        #     input_points_RV = None
        #     input_labels_RV = None
        # if all_points_MYO is not None:
        #     input_points_MYO, input_labels_MYO = np.array(all_points_MYO), np.array(all_labels_MYO)
        # else: 
        #     input_points_MYO = None   
        #     input_labels_MYO = None   


        

          




        # predictor.set_image(image)
        # masks_LV, iou_predictions_LV, lows =  predictor.predict(
        #     point_coords = input_points_LV,
        #     point_labels = input_labels_LV,
        #     multimask_output= False)


        # predictor.set_image(image)
        # masks_RV, iou_predictions_RV, lows =  predictor.predict(
        #     point_coords = input_points_RV,
        #     point_labels = input_labels_RV,
        #     multimask_output= False)
        
        # predictor.set_image(image)
        # masks_MYO, iou_predictions_MYO, lows =  predictor.predict(
        #     point_coords = input_points_MYO,
        #     point_labels = input_labels_MYO,
        #     multimask_output= False)




        # mask = np.zeros((masks_LV.shape[1], masks_LV.shape[2]), np.uint8)
        # if all_points_LV is not None:
        #     mask[masks_LV[0] != False]  = 1
        # if all_points_MYO is not None:
        #     mask[masks_MYO[0] != False] = 2
        # if all_points_RV is not None:
        #     mask[masks_RV[0] != False]  = 3
        # if all_points_MYO is None:
        #     mask = np.zeros((masks_LV.shape[1], masks_LV.shape[2]), np.uint8)

        # im = Image.fromarray(mask)
        # im.convert('L').save(f'{result_path}/{id[0][:-3]}.png')
        
    #     plt.figure(figsize=(10,10))
    #     # plt.imshow(image)
    #     plt.axis('off')
    # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #     plt.imshow(mask,cmap='gray')
    #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
    #     plt.savefig(f'data/333/{id[0]}_mask_.png', bbox_inches='tight', pad_inches=0) 
    #     plt.close()

    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image,cmap='gray')
    #     plt.axis('off')
    # #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #     plt.imshow(mask,alpha=0.5,cmap='gray')
    #     # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
    #     plt.savefig(f'data/222/{id[0]}_mask_.png', bbox_inches='tight', pad_inches=0) 
    #     plt.close()

    #     for i, j in enumerate(range(masks_LV.shape[0])):
    #         plt.figure(figsize=(10,10))
    #         plt.imshow(image)
    #         plt.axis('off')
    #     #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #         plt.imshow(masks_LV[j],alpha=0.5,cmap='gray')
    #         # plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
    #         plt.savefig(f'data/222/{id[0]}_mask_LV.png', bbox_inches='tight', pad_inches=0) 
    #         plt.close()

    #     for i, j in enumerate(range(masks_RV.shape[0])):
    #         plt.figure(figsize=(10,10))
    #         plt.imshow(image)
    #         plt.axis('off')
    #     #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #         plt.imshow(masks_RV[j],alpha=0.5,cmap='gray')
    #         # plt.title(f"Mask {i+1}, Score: {(iou_predictions_RV[j]).item():.3f}", fontsize=18)
    #         plt.savefig(f'data/222/{id[0]}_mask_RV.png', bbox_inches='tight', pad_inches=0) 
    #         plt.close()

    #     for i, j in enumerate(range(masks_MYO.shape[0])):
    #         plt.figure(figsize=(10,10))
    #         plt.imshow(image)
    #         plt.axis('off')
    #     #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
    #         plt.imshow(masks_MYO[j],alpha=0.5,cmap='gray')
    #         # plt.title(f"Mask {i+1}, Score: {(iou_predictions_MYO[j]).item():.3f}", fontsize=18)
    #         plt.savefig(f'data/222/{id[0]}_mask_MYO.png', bbox_inches='tight', pad_inches=0) 
    #         plt.close()


#         dices, HD95,IOU = calculate_metrics(torch.from_numpy(label), torch.from_numpy(mask), num_class)

#         for cls in range(num_class - 1 ):
#                 total_dice_scores[cls] += dices[cls]
#                 total_HD95_scores[cls] += HD95[cls]
#                 total_IOU_scores[cls] += IOU[cls]


    

#     avg_dice_scores = [total / num_batches for total in total_dice_scores]
#     avg_HD95_scores = [total / num_batches for total in total_HD95_scores]
#     avg_IOU_scores =  [total / num_batches for total in total_IOU_scores]

#     avg_dice = np.mean(avg_dice_scores)
#     avg_HD95 = np.mean(avg_HD95_scores)
#     avg_IOU =  np.mean(avg_IOU_scores)

#     print(f'category LV:   Dice: {avg_dice_scores[0]:.3f},  HD95: {avg_HD95_scores[0]:.3f},  IOU: {avg_IOU_scores[0]:.3f}')
#     print(f'category MYO:  Dice: {avg_dice_scores[1]:.3f},  HD95: {avg_HD95_scores[1]:.3f},  IOU: {avg_IOU_scores[1]:.3f}')
#     print(f'category RV:   Dice: {avg_dice_scores[2]:.3f},  HD95: {avg_HD95_scores[2]:.3f},  IOU: {avg_IOU_scores[2]:.3f}')

#     print(f'Total:       Dice: {avg_dice:.3f},  HD95: {avg_HD95:.3f},  IoU: {avg_IOU:.3f}')
#     empty_df = pd.DataFrame()
#     df = pd.DataFrame({  
#     # "          ": experiment_name,  
#     "Total_Dice": [avg_dice],
#     "Total_HD95": [avg_HD95],
#     "Total_IOU" : [avg_IOU],
#     "category_0_Dice": [avg_dice_scores[0]],
#     "category_0_HD95": [avg_HD95_scores[0]],
#     "category_0_IOU" : [avg_IOU_scores[0]],
#     "category_1_Dice": [avg_dice_scores[1]],
#     "category_1_HD95": [avg_HD95_scores[1]],
#     "category_1_IOU" : [avg_IOU_scores[1]],
#     "category_2_Dice": [avg_dice_scores[2]],
#     "category_2_HD95": [avg_HD95_scores[2]],
#     "category_2_IOU" : [avg_IOU_scores[2]]
# })  
#     df = df.astype(float).round(3)
#     empty_df.to_csv(f"result/val_contour_results.csv",mode = 'a', header= False, index= False)
#     df.to_csv(f"result/val_contour_results.csv",mode = 'a', header= False, index= True)
    


# # 显示LV，RV和MYO三个部分分割mask
# #         for i, j in enumerate(range(masks_LV.shape[0])):
# #             plt.figure(figsize=(10,10))
# #             plt.imshow(image)
# #             plt.axis('off')
# #         #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
# #             plt.imshow(masks_LV[j],alpha=0.5)
# #             plt.title(f"Mask {i+1}, Score: {(iou_predictions_LV[j]).item():.3f}", fontsize=18)
# #             plt.savefig(f'data/scribble/{id[0]}_{i}_masks_LV.png', bbox_inches='tight', pad_inches=0) 
# #             plt.close()

# #         for i, j in enumerate(range(masks_RV.shape[0])):
# #             plt.figure(figsize=(10,10))
# #             plt.imshow(image)
# #             plt.axis('off')
# #         #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
# #             plt.imshow(masks_RV[j],alpha=0.5)
# #             plt.title(f"Mask {i+1}, Score: {(iou_predictions_RV[j]).item():.3f}", fontsize=18)
# #             plt.savefig(f'data/scribble/{id[0]}_{i}_masks_RV.png', bbox_inches='tight', pad_inches=0) 
# #             plt.close()

# #         for i, j in enumerate(range(masks_MYO.shape[0])):
# #             plt.figure(figsize=(10,10))
# #             plt.imshow(image)
# #             plt.axis('off')
# #         #     plt.savefig(f'data/scribble/{id[0]}_image.png', bbox_inches='tight', pad_inches=0) 
# #             plt.imshow(masks_MYO[j],alpha=0.5)
# #             plt.title(f"Mask {i+1}, Score: {(iou_predictions_MYO[j]).item():.3f}", fontsize=18)
# #             plt.savefig(f'data/scribble/{id[0]}_{i}_masks_MYO.png', bbox_inches='tight', pad_inches=0) 
# #             plt.close()
            






# # 显示Scribble中点
        # label1 = label[0].cpu().numpy()
        # # label = label[0].cpu()
        # label1[label1 == 1] = 255
        # plt.imshow(image, cmap='gray')
        # plt.imshow(label1, cmap='gray', alpha= 0.5)
        
        # if input_points_LV is not None:
        #     for i in range(input_points_LV.shape[0]):
        #         x, y  = input_points_LV[i]
        #         if input_labels_LV[i]==1:
        #             plt.plot(x, y, 'ro')
        #         else: plt.plot(x, y, 'go')

                
        # plt.axis('off')
        # plt.savefig(f'data/111/{id[0]}_label_LV.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        
        
        # label1 = label[0].cpu().numpy()
        # label1[label1 == 3] = 255
        # plt.imshow(image, cmap='gray')
        # plt.imshow(label1, cmap='gray', alpha= 0.5)
        # if input_points_RV is not None:
        #     for i in range(input_points_RV.shape[0]):
        #         x, y  = input_points_RV[i]
        #         if input_labels_RV[i]==1:
        #             plt.plot(x, y, 'ro')
        #         else: plt.plot(x, y, 'go')
        # plt.axis('off')
        # plt.savefig(f'data/111/{id[0]}_label_RV.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        

        # label1 = label[0].cpu().numpy()
        # label1[label1 == 2] = 255
        # plt.imshow(image, cmap='gray')
        # plt.imshow(label1, cmap='gray', alpha= 0.5)
        # if input_points_MYO is not None:
        #     for i in range(input_points_MYO.shape[0]):
        #         x, y  = input_points_MYO[i]
        #         if input_labels_MYO[i]==1:
        #             plt.plot(x, y, 'ro')
        #         else: plt.plot(x, y, 'go')
        # plt.axis('off')
        # plt.savefig(f'data/111/{id[0]}_label_MYO.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
