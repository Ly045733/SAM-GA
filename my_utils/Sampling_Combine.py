import numpy as np
import torch

####Sampling

# 从给出的scibble中  按着轮廓  采样出指定类别、指定数目的点
def contour_sample(scribble, ind, num):
    sampled_point_batch = []
    for batch in range(scribble.shape[0]):
        points = np.column_stack(np.where(scribble[batch].cpu() == ind))
        points[:, [0, 1]] = points[:, [1, 0] ]
        if len(points) == 0:
                return None
        else:
            if num >= len(points):
                sampled_points = points 
                sampled_point_batch.append(sampled_points)  
            else:
                indices = np.linspace(0, len(points) - 1, num, dtype=int)
                sampled_points = points[indices]
                sampled_point_batch.append(sampled_points)       

    return sampled_point_batch

def contour_sample_without_bs(scribble, ind, num):
    sampled_point_batch = []
    points = np.column_stack(np.where(scribble.cpu() == ind))
    points[:, [0, 1]] = points[:, [1, 0] ]
    if len(points) == 0:
            return None
    else:
        if num >= len(points):
            sampled_points = points 
            sampled_point_batch.append(sampled_points)  
        else:
            indices = np.linspace(0, len(points) - 1, num, dtype=int)
            sampled_points = points[indices]
            sampled_point_batch.append(sampled_points)       

    return sampled_point_batch

# 从给出的scibble中  随机  采样出指定类别、指定数目的点
def random_sample(scribble, ind, num):
    sampled_point_batch = []
    for batch in range(scribble.shape[0]):
        points = np.column_stack(np.where(scribble[batch].cpu() == ind))
        points[:, [0, 1]] = points[:, [1, 0] ]
        if len(points) == 0:
                return None
        else:
            if num>= len(points):
                sampled_points = points 
                sampled_point_batch.append(sampled_points)  
            else:
                indices = np.random.uniform(0, len(points) - 1, num).astype(np.int8())
                sampled_points = points[indices]
                sampled_point_batch.append(sampled_points)       

    return sampled_point_batch

#划分涂鸦区域为网格
def split_into_grids(coords, image, grid_num, image_size):
    grids = []
    h, w = image_size
    all_grids = 0.0
    X_max, X_min, Y_max, Y_min = np.max(coords[:, 0]), np.min(coords[:, 0]), np.max(coords[:, 1]), np.min(coords[:, 1])
    grid_size = max(((X_max - X_min)/grid_num).astype(np.int8())+1,  ((Y_max - Y_min)/grid_num).astype(np.int8())+1)
    if Y_min==Y_max:
        for x_start in range(X_min, X_max, grid_size):
            x_end = min(x_start + grid_size, w)
            grid_mask = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end)
            grid_coords = coords[grid_mask]

            bias = int(grid_size/2)
            grid_image_all = np.sum(image[Y_min-bias:Y_min+bias, x_start:x_end])
            if grid_coords.shape[0] !=0 :
                all_grids += grid_image_all
            if len(grid_coords) > 0:
                grids.append(grid_coords)
    if X_min==X_max:
        for y_start in range(Y_min, Y_max, grid_size):
            y_end = min(y_start + grid_size, h) 
            grid_mask = (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
            grid_coords = coords[grid_mask]

            bias = int(grid_size/2)
            grid_image_all = np.sum(image[y_start:y_end, X_min-bias:X_min+bias])
            if grid_coords.shape[0] !=0 :
                all_grids += grid_image_all
            if len(grid_coords) > 0:
                grids.append(grid_coords)

    else:
        for y_start in range(Y_min, Y_max, grid_size):
            for x_start in range(X_min, X_max, grid_size):
                # 获取网格区域的边界
                x_end = min(x_start + grid_size, w)
                y_end = min(y_start + grid_size, h)
                
                # 获取当前网格内所有像素点的坐标
                grid_mask = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
                grid_coords = coords[grid_mask]

                grid_image_all = np.sum(image[y_start:y_end, x_start:x_end])

                if grid_coords.shape[0] !=0 :
                    all_grids += grid_image_all
                # 如果网格内有像素点，加入到网格列表
                if len(grid_coords) > 0:
                    grids.append(grid_coords)
    if grids==[]:
        print(1) 
    return grids,all_grids


def cross_entropy(p, q):
    # 避免log(0)的情况，可以做一个小的数值平移
    epsilon = 1e-10
    q = np.clip(q, epsilon, 1. - epsilon)
    return -(p * np.log(q))

def select_pixel(image, points_grid, image_all, small_grid_all, image_size):
    grid_image_all = small_grid_all
    if grid_image_all==0:
        grid_image_all = 1
    points_all = np.zeros(shape = (points_grid.shape[0]),dtype=image.dtype)
    h, w = image_size
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            y = np.minimum(points_grid[:, 1]+i,h-1)  
            x = np.minimum(points_grid[:, 0]+j,w-1)      
            Px = image[y, x]
            points_all += Px
    points_all =  points_all/grid_image_all
    Entropy = cross_entropy(points_all, points_all)
    min_idx = np.argmax(Entropy)
    return Entropy[min_idx],points_grid[min_idx]
    



def Entropy_Grids_Sampling(scribble, image,  ind, num):
    sampled_point_batch = []
    image_all = torch.sum(image).detach().cpu().numpy().item()
    for batch in range(scribble.shape[0]):
        points = np.column_stack(np.where(scribble[batch].cpu() == ind))
        points[:, [0, 1]] = points[:, [1, 0] ]
        if len(points) == 0:
                return None
        else:
            if num >= len(points):
                sampled_points = points 
                sampled_point_batch.append(sampled_points)  
            else:
                E_max_grid, indices = [], [] 
                point_grids, small_grid_all= split_into_grids(points, image[batch][0].cpu().numpy(), num, scribble[batch].shape)
                for points_grid in point_grids:
                    E_max, point = select_pixel(image[batch][0].cpu().numpy(), points_grid, image_all, small_grid_all, scribble[batch].shape)
                    E_max_grid.append(E_max)
                    indices.append(point)
                top_num_idx = np.array(E_max_grid).argsort()[-1:-(num+1):-1]  
                sampled_points = np.array(indices)[top_num_idx, :]   
                sampled_point_batch.append(sampled_points) 
    return sampled_point_batch 


def select_pixel_contour(image, indices, image_all, scribble_all, image_size):
    big_foundation, small_foundation = image_all, scribble_all
    points_all = np.zeros(shape = (indices.shape[0]),dtype=image.dtype)
    h, w = image_size
    if small_foundation == 0:
        small_foundation = 1
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            if i==0 & j==0:
                continue 
            y = np.minimum(indices[:, 1]+i,h-1)  
            x = np.minimum(indices[:, 0]+j,w-1)      
            Px = image[y, x]
            points_all += Px
    points_all =  points_all/small_foundation
    Entropy = cross_entropy(points_all, points_all)
    max_idx = np.argmax(Entropy)
    return indices[max_idx]


def Entropy_contour_Sampling(scribble, image,  ind, num):
    sampled_point_batch = []
    image_all = torch.sum(image).detach().cpu().numpy().item()
    points = np.column_stack(np.where(scribble.cpu() == ind))
    points[:, [0, 1]] = points[:, [1, 0] ]
    image = image[0,:,:].cpu().numpy()
    if len(points) == 0:
            return None        
    else:
        X_min,X_max,Y_min,Y_max = min(points[:, 0]),max(points[:, 0]),min(points[:, 1]),max(points[:, 1])
        scribble_all = np.sum(image[Y_min:Y_max, X_min:X_max]).item()
        if num >= len(points):
            sampled_points = points 
            sampled_point_batch.append(sampled_points)  
        else:
            indices_all = []
            indices = np.linspace(0, len(points) - 1, num+1, dtype=int)
            for i in range(num):
                coor = points[indices[i]:indices[i+1],:]
                if i == num:
                        coor = points[indices[i]:indices[i+1]+1,:]
                

                point = select_pixel_contour(image,coor,image_all,scribble_all,scribble.shape) 
                indices_all.append(point) 
            sampled_point_batch.append(indices_all) 
    return sampled_point_batch 

    # indices = np.linspace(0, len(points) - 1, num, dtype=int)
    # sampled_points = points[indices]
    # sampled_point_batch.append(sampled_points)

####Combine

# 将采样得到的各个类别的点按着正负类组合起来并返回组合后的点以及对应类别
def combine(point_LV, point_RV, point_MYO, point_BackGrond):
    all_points_LV,  all_labels_LV = [], []
    all_points_RV,  all_labels_RV = [], []
    all_points_MYO, all_labels_MYO= [], []
    if point_LV is not None:
        all_points_LV.extend(point_LV[0])
        all_labels_LV.extend([1]*len(point_LV[0]))
        if point_MYO is not None:
            all_points_LV.extend(point_MYO[0])
            all_labels_LV.extend([0]*len(point_MYO[0]))
        elif point_RV is not None:
            all_points_LV.extend(point_RV[0])
            all_labels_LV.extend([0]*len(point_RV[0]))  
        else:
            all_points_LV.extend(point_BackGrond[0])
            all_labels_LV.extend([0]*len(point_BackGrond[0]))
    else:
        all_points_LV = None


    if point_RV is not None:
        all_points_RV.extend(point_RV[0])
        all_labels_RV.extend([1]*len(point_RV[0]))
        if point_MYO is not None:
            all_points_RV.extend(point_MYO[0])
            all_labels_RV.extend([0]*len(point_MYO[0]))
        elif point_LV is not None:
            all_points_RV.extend(point_LV[0])
            all_labels_RV.extend([0]*len(point_LV[0]))  
        else:
            all_points_RV.extend(point_BackGrond[0])
            all_labels_RV.extend([0]*len(point_BackGrond[0]))
    else:
        all_points_RV = None

        
    if point_MYO is not None:
        all_points_MYO.extend(point_MYO[0])
        all_labels_MYO.extend([1]*len(point_MYO[0]))
        if point_LV is not None:
            all_points_MYO.extend(point_LV[0])
            all_labels_MYO.extend([0]*len(point_LV[0]))
        elif point_RV is not None:
            all_points_MYO.extend(point_RV[0])
            all_labels_MYO.extend([0]*len(point_RV[0]))  
        else:
            all_points_MYO.extend(point_BackGrond[0])
            all_labels_MYO.extend([0]*len(point_BackGrond[0])) 
    else:
        all_points_MYO = None

    return all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO


def combine_cell(point_Nu, point_Cy, point_BackGrond):
    all_points_Nu,  all_labels_Nu = [], []
    all_points_Cy,  all_labels_Cy = [], []
    if point_Nu is not None:
        all_points_Nu.extend(point_Nu[0])
        all_labels_Nu.extend([1]*len(point_Nu[0]))
        if point_Cy is not None:
            all_points_Nu.extend(point_Cy[0])
            all_labels_Nu.extend([0]*len(point_Cy[0]))
        else:
            all_points_Nu.extend(point_BackGrond[0])
            all_labels_Nu.extend([0]*len(point_BackGrond[0]))
    else:
        all_points_LV = None


    if point_Cy is not None:
        all_points_Cy.extend(point_Cy[0])
        all_labels_Cy.extend([1]*len(point_Cy[0]))
        if point_BackGrond is not None:
            all_points_Cy.extend(point_BackGrond[0])
            all_labels_Cy.extend([0]*len(point_BackGrond[0]))
        elif point_Nu is not None:
            all_points_Cy.extend(point_Nu[0])
            all_labels_Cy.extend([0]*len(point_Nu[0]))  
    else:
        all_points_RV = None

    return all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy
# 按着batch来组合内容，组合后的内容服从Sam输入
def process_input_SAM(transform,image_RGB,original_size, all_points_LV, all_labels_LV, all_points_RV, all_labels_RV, all_points_MYO, all_labels_MYO, batched_input_LV, batched_input_MYO, batched_input_RV):
    if all_points_LV is None :
                batched_input_LV.append(
                {
                    'image': image_RGB,
                    'original_size': original_size
                }
        )
    else:
        all_points_LV = transform.apply_coords(np.array(all_points_LV),original_size)
        all_points_LV = torch.as_tensor(all_points_LV, dtype=torch.float, device='cuda').reshape(len(all_labels_LV),2)
        all_labels_LV = torch.as_tensor(all_labels_LV, dtype=torch.float, device='cuda').reshape(len(all_labels_LV))
        all_points_LV = all_points_LV[None, :, :]
        all_labels_LV = all_labels_LV[None, :] 
        batched_input_LV.append(
            {
                'image': image_RGB,
                'point_coords': all_points_LV,
                'point_labels': all_labels_LV,
                'original_size': original_size
            }
        )
        
    if all_points_RV is None:
        batched_input_RV.append(
        {
            'image': image_RGB,
            'original_size': original_size
        }
    )
        
    else:
        all_points_RV = transform.apply_coords(np.array(all_points_RV),original_size)
        all_points_RV = torch.as_tensor(all_points_RV, dtype=torch.float, device='cuda').reshape(len(all_labels_RV),2)
        all_labels_RV = torch.as_tensor(all_labels_RV, dtype=torch.float, device='cuda').reshape(len(all_labels_RV))
        all_points_RV = all_points_RV[None, :, :] 
        all_labels_RV = all_labels_RV[None, :] 
        batched_input_RV.append(
            {
                'image': image_RGB,
                'point_coords': all_points_RV,
                'point_labels': all_labels_RV,
                'original_size': original_size
            }
        )

    if all_points_MYO is None:
        batched_input_MYO.append(
        {
            'image': image_RGB,
            'original_size': original_size
        }
    )
        
    else:
        all_points_MYO = transform.apply_coords(np.array(all_points_MYO),original_size)
        all_points_MYO = torch.as_tensor(all_points_MYO, dtype=torch.float, device='cuda').reshape(len(all_labels_MYO),2)
        all_labels_MYO = torch.as_tensor(all_labels_MYO, dtype=torch.float, device='cuda').reshape(len(all_labels_MYO))
        
        
        all_points_MYO = all_points_MYO[None, :, :]
        all_labels_MYO = all_labels_MYO[None, :]
    
        
        
        batched_input_MYO.append(
            {
                'image': image_RGB,
                'point_coords': all_points_MYO,
                'point_labels': all_labels_MYO,
                'original_size': original_size
            }
        )
    return batched_input_LV, batched_input_MYO, batched_input_RV

def process_input_SAM_cell(transform,image_RGB,original_size, all_points_Nu, all_labels_Nu, all_points_Cy, all_labels_Cy, batched_input_Nu, batched_input_Cy):
    if all_points_Nu is None :
                batched_input_Nu.append(
                {
                    'image': image_RGB,
                    'original_size': original_size
                }
        )
    else:
        all_points_Nu = transform.apply_coords(np.array(all_points_Nu),original_size)
        all_points_Nu = torch.as_tensor(all_points_Nu, dtype=torch.float, device='cuda').reshape(len(all_labels_Nu),2)
        all_labels_Nu = torch.as_tensor(all_labels_Nu, dtype=torch.float, device='cuda').reshape(len(all_labels_Nu))
        all_points_Nu = all_points_Nu[None, :, :]
        all_labels_Nu = all_labels_Nu[None, :] 
        batched_input_Nu.append(
            {
                'image': image_RGB,
                'point_coords': all_points_Nu,
                'point_labels': all_labels_Nu,
                'original_size': original_size
            }
        )
        
    if all_points_Cy is None:
        batched_input_Cy.append(
        {
            'image': image_RGB,
            'original_size': original_size
        }
    )
        
    else:
        all_points_Cy = transform.apply_coords(np.array(all_points_Cy),original_size)
        all_points_Cy = torch.as_tensor(all_points_Cy, dtype=torch.float, device='cuda').reshape(len(all_labels_Cy),2)
        all_labels_Cy = torch.as_tensor(all_labels_Cy, dtype=torch.float, device='cuda').reshape(len(all_labels_Cy))
        all_points_Cy = all_points_Cy[None, :, :] 
        all_labels_Cy = all_labels_Cy[None, :] 
        batched_input_Cy.append(
            {
                'image': image_RGB,
                'point_coords': all_points_Cy,
                'point_labels': all_labels_Cy,
                'original_size': original_size
            }
        )

    
    return batched_input_Nu, batched_input_Cy




# 按着batch来组合内容，组合后的内容服从Sam输入。
def process_input(batch_img, batch_point, tranformer, original_size):
    batch_inputs = []
    batch_labels, batch_points = [], []
    flag = 0
    for label, (class_name, points) in enumerate(batch_point.items()):                
        if class_name =='background':
                for ba in range(batch_img.shape[0]):
                        if flag == 0:
                                batch_labels.extend([[0] * len(points[ba])])
                                batch_points.extend([points[ba]])
                        else: 
                                batch_labels[ba].extend([0] * len(points[ba]))
                                batch_points[ba] = np.vstack((batch_points[ba], points[ba]))                                
                flag = 1
        else:
                for ba in range(batch_img.shape[0]):
                        if flag == 0:
                                batch_labels.extend([[1] * len(points[ba])])
                                batch_points.extend([points[ba]])
                        else: 
                                batch_labels[ba].extend([1] * len(points[ba]))
                                batch_points[ba] = np.vstack((batch_points[ba], points[ba]))                                
                flag = 1
    for ba in range(batch_img.shape[0]):
        batch_input = {
                'image' : batch_img[ba].cuda(),
                'point_coords' : torch.tensor(tranformer.apply_coords(batch_points[ba], original_size)).unsqueeze(1).cuda(),
                'point_labels' : torch.tensor(batch_labels[ba]).unsqueeze(1).cuda(),
                'original_size' : (original_size[0], original_size[1]),
        }
        batch_inputs.append(batch_input)
    return batch_inputs
