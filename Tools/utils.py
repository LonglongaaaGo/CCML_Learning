import os
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import shutil

def save_opt(args):
    root_name = './checkpoint/' + args.filename
    args = vars(args)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    save_name = "opt.txt"
    file_name = os.path.join(root_name, save_name)

    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')




def get_thresholds(bins=1,interval=(0.5,0.5)):
    """
    :param bins: the number of the threshold
    :param interval: give the min number and max number for a interval
    :return: the list that meet the conditions
    """
    max_iter = interval[1]-interval[0]
    each_iter = float(max_iter)/bins
    threshold = interval[0]
    threshold_list = []
    for i in range(bins):
        threshold_list.append(threshold)
        threshold+=each_iter
    threshold_list.append(interval[1])
    return threshold_list

def get_IOUs(pre_masks, gt_mask,threshold_list=[]):
    gt_mask = F.interpolate(gt_mask, (pre_masks.shape[2], pre_masks.shape[3]))

    thre_IOU_dic = {}
    for threshold in threshold_list:
        temp_pre_masks = (pre_masks > threshold).int().cpu()

        area_intersection, area_union = intersectionAndUnion(imPred=temp_pre_masks, imLab=gt_mask, numClass=1,type="ignore_0")
        area_inter_acc = area_intersection.sum()
        area_union_acc = area_union.sum()

        thre_IOU_dic[threshold] = [area_inter_acc,area_union_acc]
    return thre_IOU_dic




def intersectionAndUnion(imPred, imLab, numClass,type="contain_unlabeled"):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    if type == "contain_unlabeled":
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred += 1
        imLab += 1
    else:
        ##  Ignore IoU for background class ("0")
        pass
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    #bins:统计的区间个数
    #range是一个长度为2的元组，表示统计范围的最小值和最大值，默认值None，表示范围由数据的范围决定
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def debug_show(input_tensor,type_="bi_mask"):
    import matplotlib.pyplot as plt
    img = convert_image_np(input_tensor[0].cpu().detach(),type_=type_)
    plt.imshow(img,vmin=0,vmax=1)
    plt.show()


def visual_for_CCML(imgs,masks,predicted_masks,img_paths,out_path):

    mask_out = os.path.join(out_path,"mask")
    makedirs(mask_out)
    predict_mask_out = os.path.join(out_path, "predict_mask")
    makedirs(predict_mask_out)

    predict_mask_out0_5 = os.path.join(out_path, "predict_mask0_5")
    makedirs(predict_mask_out0_5)

    predict_mask_img_out = os.path.join(out_path, "predict_mask_img")
    makedirs(predict_mask_img_out)

    mask_img_out = os.path.join(out_path, "mask_img")
    makedirs(mask_img_out)

    predict_mask_img0_5 = os.path.join(out_path, "predict_mask_img0_5")
    makedirs(predict_mask_img0_5)

    # masks = F.adaptive_avg_pool2d(masks, (predicted_masks.size(2), predicted_masks.size(3)))
    masks = F.interpolate(masks, (predicted_masks.size(2), predicted_masks.size(3)), mode="nearest")
    feature_map_size = imgs.size()

    for i in range(imgs.shape[0]):
        # predicted_masks[i] = predicted_masks[i]>0.5
        predic_mask = convert_image_np(predicted_masks[i].cpu().detach(),type_="bi_mask")
        mask = convert_image_np(masks[i].cpu().detach(),type_="bi_mask")
        predic_mask_0_5 = predicted_masks[i].clone()>0.5
        predic_mask_0_5 = convert_image_np(predic_mask_0_5.cpu().detach(),type_="bi_mask")


        raw_img = Image.open(img_paths[i]).convert("RGB")
        raw_img = raw_img.resize((feature_map_size[2],feature_map_size[3]))
        raw_img = np.asarray(raw_img)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)


        predic_mask_0_5_color = predic_mask_0_5.copy()

        # only get red channel
        predic_mask_0_5_color[:, :, 0] = 0
        predic_mask_0_5_color[:, :, 2] = 0

        predic_mask_0_5_color = cv2.resize(predic_mask_0_5_color, (feature_map_size[3], feature_map_size[2]),
                                       interpolation=cv2.INTER_NEAREST)


        raw_predic_mask_0_5_color= cv2.addWeighted(raw_img.astype(np.uint8), 0.5, predic_mask_0_5_color.astype(np.uint8), 0.5,
                                                 0)

        predic_mask_0_5 = cv2.resize(predic_mask_0_5, (feature_map_size[3], feature_map_size[2]),
                                 interpolation=cv2.INTER_NEAREST)




        # cv2.imwrite('ImageNet/Visualization/DDT/VGG16-vis-test_3/test.jpg', atten_norm.astype(np.uint8))
        # predic_mask = cv2.resize(predic_mask, (feature_map_size[3],feature_map_size[2]), interpolation=cv2.INTER_NEAREST)
        predic_mask_color = predic_mask.copy()
        # only get red channel
        predic_mask_color[:,:,0] = 0
        predic_mask_color[:,:,2] = 0

        predic_mask_color = cv2.resize(predic_mask_color, (feature_map_size[3],feature_map_size[2]), interpolation=cv2.INTER_NEAREST)
        # min_val = np.min(predic_mask_color)
        # max_val = np.max(predic_mask_color)
        # atten_norm = (predic_mask_color - min_val) / (max_val - min_val)
        # atten_norm = atten_norm * 255
        # heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)

        raw_predicted_mask_img = cv2.addWeighted(raw_img.astype(np.uint8), 0.5, predic_mask_color.astype(np.uint8), 0.5, 0)

        predic_mask[:] = predic_mask[:]
        predic_mask = cv2.resize(predic_mask, (feature_map_size[3], feature_map_size[2]),
                                 interpolation=cv2.INTER_NEAREST)


        #cv2.INTER_LINEAR
        # only get red channel
        mask[:, :, 0] = 0
        mask[:, :, 2] = 0
        mask = cv2.resize(mask, (feature_map_size[3],feature_map_size[2]), interpolation=cv2.INTER_NEAREST)
        heat_map_mask = mask
        # mask = cv2.resize(mask, (feature_map_size[3],feature_map_size[2]), interpolation=cv2.INTER_LINEAR)
        # min_val = np.min(mask)
        # max_val = np.max(mask)
        # atten_norm_mask = (mask - min_val) / (max_val - min_val)
        # atten_norm_mask = atten_norm_mask * 255
        # heat_map_mask = cv2.applyColorMap(heat_map_mask.astype(np.uint8), cv2.COLORMAP_JET)
        raw_mask_img = cv2.addWeighted(raw_img.astype(np.uint8), 0.5, heat_map_mask.astype(np.uint8), 0.5, 0)

        img_paths_split = img_paths[i].split("/")
        category_name = img_paths_split[-2]
        name = img_paths_split[-1]

        mask_path = os.path.join(mask_out, category_name)
        makedirs(mask_path)
        mask_path = os.path.join(mask_path,name)
        mask_img_path = os.path.join(mask_img_out, category_name)
        makedirs(mask_img_path)

        mask_img_path =  os.path.join(mask_img_path,name)

        mask_0_5_path = os.path.join(predict_mask_out0_5, category_name)
        makedirs(mask_0_5_path)
        mask_0_5_path = os.path.join(mask_0_5_path, name)
        predict_mask_img0_5_path = os.path.join(predict_mask_img0_5, category_name)
        makedirs(predict_mask_img0_5_path)
        predict_mask_img0_5_path = os.path.join(predict_mask_img0_5_path, name)


        predict_mask_path = os.path.join(predict_mask_out, category_name)
        makedirs(predict_mask_path)

        predict_mask_path =  os.path.join(predict_mask_path,name)
        predict_mask_img_path = os.path.join(predict_mask_img_out, category_name)
        makedirs(predict_mask_img_path)
        predict_mask_img_path =  os.path.join(predict_mask_img_path,name)


        cv2.imwrite(mask_path, np.asarray(mask))
        cv2.imwrite(mask_img_path, np.asarray(raw_mask_img))
        cv2.imwrite(predict_mask_path, np.asarray(predic_mask))
        cv2.imwrite(predict_mask_img_path, np.asarray(raw_predicted_mask_img))

        cv2.imwrite(mask_0_5_path, np.asarray(predic_mask_0_5))
        cv2.imwrite(predict_mask_img0_5_path, np.asarray(raw_predic_mask_0_5_color))



def convert_image_np(inp,type_ = "std",mean =[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]):
    """Convert a Tensor to numpy image."""
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # if inp.shape[0] == 1:
    #     inp = inp.numpy()
    # elif inp.shape[0] == 3:
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    if type_ == "std":
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    elif type_ == "bi_mask":
        inp = 255.0 * inp
        from PIL import Image
        Binary_map = inp
        # Binary_map = np.expand_dims(np.asarray(inp), 2)
        temp_Binary_map = np.concatenate((Binary_map.copy(), Binary_map.copy()), axis=2)
        inp = np.concatenate((temp_Binary_map.copy(), Binary_map), axis=2)
        #
        inp = inp.astype(np.uint8)

        # inp = Image.fromarray(Binary_map, mode='RGB')

        return inp
    else :
        print("error! convert_image_np" )



def vis_files(file_list,output_root="error_out"):
    print("output error example")

    if os.path.exists(output_root):
        shutil.rmtree(output_root)  # delete output folder
    os.makedirs(output_root)  # make new output folder

    for i in range(len(file_list)):
        shutil.copy(file_list[i],output_root)

    print("down!")




def check_data_parallel(checkpoint):
    for k, v in checkpoint['model'].items():
        if 'module.' in k: return True
    return False