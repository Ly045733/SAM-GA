from Networks.unet import UNet, UNet_DS, UNet_CCT, UNet_TRI ,UNet_TRI_ATT,UNet_CCT_SlotAttention,UNet_CCT_SlotAttention_thin,UNet_CCT_SlotAttention_thin_aux,UNet_CCT_SlotAttention_thin_edge
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_SA":
        net = UNet_CCT_SlotAttention(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_SA_thin":
        net = UNet_CCT_SlotAttention_thin(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_SA_thin_aux":
        net = UNet_CCT_SlotAttention_thin_aux(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_SA_thin_edeg":
        net = UNet_CCT_SlotAttention_thin_edge(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_tri":
        net = UNet_TRI(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_tri_att":
        net = UNet_TRI_ATT(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
        print(f"{net_type} is invalid")
    return net
