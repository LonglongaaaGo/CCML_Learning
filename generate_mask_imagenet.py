import os
import sys
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from skimage import measure
# from scipy.misc import imresize
from utils.func import *
from utils.vis import *
from utils.IoU import *
import argparse
from loader.ddt_imagenet_dataset import DDTImageNetDataset


parser = argparse.ArgumentParser(description='Parameters for DDT generate mask')
parser.add_argument('--input_size',default=512,dest='input_size')
parser.add_argument('--data',metavar='DIR',help='path to your target dataset')
parser.add_argument('--gpu',help='which gpu to use',default='0,1',dest='gpu')
parser.add_argument('--output_path',default='ImageNet/Projection/VGG16-448',dest='output_path for box ')
parser.add_argument('--output_mask_path',default='/home/image/Data/all_mask',dest='output_mask_path')

parser.add_argument('--batch_size',default=32,dest='batch_size')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"
cudnn.benchmark = True
model_ft = models.vgg16(pretrained=True)
model = model_ft.features
#removed = list(model.children())[:-1]
#model = torch.nn.Sequential(*removed)
model = torch.nn.DataParallel(model).cuda()
model.eval()
projdir = args.output_path
if not os.path.exists(projdir):
    os.makedirs(projdir)

transform = transforms.Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
batch_size = args.batch_size
# a = DDTImageNetDataset(root=os.path.join(args.data,'train'),batch_size=args.batch_size, transforms=transform)
a = DDTImageNetDataset(root=os.path.join(args.data,''),batch_size=args.batch_size, transforms=transform)

# print(classes[0])

# args.output_mask_path = "/home/image/DataDisk/Longlongaaago/Data/sv_data_by_logo_0811/all_mask"

for class_ind in range(1000):
    #if class_ind == 10:
    #    import sys
    #    sys.exit()
    now_class_dict = {}
    feature_list = []
    ddt_bbox = {}
    final_h = 0
    final_w = 0
    with torch.no_grad():
        for (input_img,path) in a[class_ind]:
            input_img = to_variable(input_img)
            in_B,in_c,in_h,in_w  = input_img.size()
            output = model(input_img)
            output = to_data(output)
            output = torch.squeeze(output).numpy()
            if len(output.shape) == 3:
                output = np.expand_dims(output,0)
            output = np.transpose(output,(0,2,3,1))
            n,h,w,c = output.shape
            final_h = h
            final_w = w
            for i in range(n):
                now_class_dict[path[i]] = output[i,:,:,:]
            output = np.reshape(output,(n*h*w,c))
            feature_list.append(output)
        X = np.concatenate(feature_list,axis=0)
        mean_matrix = np.mean(X, 0)
        X = X - mean_matrix
        print("Before PCA")
        trans_matrix = sk_pca(X, 1)
        print("AFTER PCA")
        cls = a.label_class_dict[class_ind]
        # save json
        d = {'mean_matrix': mean_matrix.tolist(), 'trans_matrix': trans_matrix.tolist()}
        with open(os.path.join(projdir, '%s_trans.json' % cls), 'w') as f:
            json.dump(d, f)
        # load json
        with open(os.path.join(projdir, '%s_trans.json' % cls), 'r') as f:
            t = json.load(f)
            mean_matrix = np.array(t['mean_matrix'])
            trans_matrix = np.array(t['trans_matrix'])

        print('trans_matrix shape is {}'.format(trans_matrix.shape))

        cnt = 0
        for k,v in now_class_dict.items():
            w = final_w
            h = final_h
            he = in_h
            wi = in_w
            range_ = he
            v = np.reshape(v,(h * w,512))
            v = v - mean_matrix

            heatmap = np.dot(v, trans_matrix.T)
            heatmap = np.reshape(heatmap, (h, w))
            highlight = np.zeros(heatmap.shape)
            highlight[heatmap > 0] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1

            # visualize heatmap
            # show highlight in origin image
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (wi, he), interpolation=cv2.INTER_NEAREST)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                #print(highlight)
                bbox = [0, 0, w, h]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            temp_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            temp_save_box = [x / range_ for x in temp_bbox]
            ddt_bbox[os.path.join(cls, k)] = temp_save_box

            b_highlight_big = np.asarray(highlight_big)
            highlight_big = np.expand_dims(np.asarray(highlight_big), 2)
            highlight_3 = np.concatenate((np.zeros((he, wi, 1)), np.zeros((he, wi, 1))), axis=2)
            highlight_3 = np.concatenate((highlight_3, highlight_big), axis=2)
            min_val = np.min(highlight_3)
            max_val = np.max(highlight_3)
            atten_norm = (highlight_3 - min_val) / (max_val - min_val)
            atten_norm = atten_norm * 255
            # plt.imshow('inp',np.asarray(atten_norm).astype(np.uint8))

            # if cnt < 1000000 and cnt%100==0:
            savepath = 'ImageNet/Visualization/DDT/VGG16-vis_224_sv_data_by_logo/%s' % cls
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            from PIL import Image
            raw_img = Image.open(k).convert("RGB")
            raw_img = raw_img.resize((he,wi))
            raw_img = np.asarray(raw_img)
            raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)

            # cv2.imwrite('ImageNet/Visualization/DDT/VGG16-vis-test_3/test.jpg', atten_norm.astype(np.uint8))
            raw_img = cv2.addWeighted(raw_img.astype(np.uint8), 0.5, atten_norm.astype(np.uint8), 0.5, 0)

            cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]),
                          (temp_bbox[2] + temp_bbox[0], temp_bbox[3] + temp_bbox[1]), (255, 0, 0), 4)
            save_name = k.split('/')[-1]
            # cv2.imwrite(os.path.join(savepath, save_name), np.asarray(raw_img))

            cls_path = os.path.join(args.output_mask_path, cls)
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            name_path = os.path.join(cls_path, save_name)


            #get color map
            # atten_norm = np.asarray(atten_norm)
            # atten_norm = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
            #cv2.imwrite(name_path, np.asarray(atten_norm))


            #get Binary_map
            Binary_map = np.expand_dims(np.asarray(b_highlight_big), 2)
            temp_Binary_map = np.concatenate((Binary_map.copy(), Binary_map.copy()), axis=2)
            Binary_map = np.concatenate((temp_Binary_map.copy(), Binary_map), axis=2)

            Binary_map = Binary_map.astype(np.uint8)

            img = Image.fromarray(Binary_map,mode='RGB')
            # img.show()
            img = img.convert('1')
            # img.show()
            # bb = img.resize((7,7))
            # bb.show()

            img.save(name_path)




    with open(os.path.join(projdir, '%s_bbox.json' % cls), 'w') as fp:
        json.dump(ddt_bbox, fp)