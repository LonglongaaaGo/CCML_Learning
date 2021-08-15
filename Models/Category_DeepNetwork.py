from Dataset.preprocess import load_data
from Models.basemodel import BaseModel
from Models.VLF_net import VLF_net,CCML_VLF_net
import torch
import torch.optim as optim
import torch.nn as nn
from Tools.utils import get_IOUs,get_thresholds,visual_for_CCML,makedirs,vis_files
from tqdm import tqdm
import torch.nn.functional as F
import time
import os

class Classification_Network(BaseModel):
    def __init__(self,args,checkpoint_path,type="train"):
        super().__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:"+str(args.gpus) if use_cuda else "cpu")
        self.args = args
        print(self.args)
        self.checkpoint_path = checkpoint_path

        # model
        self.model = self.getModel(modelName=args.modelName, num_classes=args.num_classes)
        print(self.model)

        # data loaders
        self.train_loader, self.test_loader, self.val_loader = self.load_datasets(self.args)

        #optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        #show and save information


    def getModel(self,modelName="VLF_net", num_classes=30):
        if modelName == "VLF_net":
            return VLF_net(num_classes=num_classes, small_inputs=True).to(self.device)

    def forward(self,data):
        data = data.to(self.device)
        output = self.model(data)
        return output

    def load_datasets(self,args):
        sizes_str = args.size
        sizes_ = sizes_str.split(",")
        siezs = [int(sizes_[i]) for i in range(len(sizes_))]

        crop_str = args.crop
        crops_ = crop_str.split(",")
        crops = [int(crops_[i]) for i in range(len(crops_))]

        args.train_dir = args.train_Root + "/train"
        args.val_dir = args.train_Root + "/test"
        args.test_dir = args.train_Root + "/test"

        if (not args.use_val == True):
            args.val_dir = args.test_dir

        train_loader, test_loader, val_loader = load_data(args, crop_height=crops[0], crop_width=crops[1]
                                                          , height=siezs[0], width=siezs[1],
                                                          num_workers=args.num_workers,dataset_mode=args.dataset_mode)
        return train_loader, test_loader, val_loader

    def train(self,epoch):
        self.model.train()
        step = 0
        train_loss = 0
        train_acc = 0
        count = 0
        acc_all = 0
        for data, target, path in tqdm(self.train_loader, desc="epoch " + str(epoch), mininterval=1):
            if self.args.debug == True and count > 100: break
            count += 1
            self.adjust_learning_rate(self.optimizer, epoch, self.args)
            self.optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)

            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            y_pred = output.data.max(1)[1]

            acc = float(y_pred.eq(target.data).sum()) * 100.

            train_acc += acc
            step += 1
            if step % 100 == 0:
                print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc / len(data)), end='')
                for param_group in self.optimizer.param_groups:
                    print(",  Current learning rate is: {}".format(param_group['lr']))

        length = len(self.train_loader.dataset)
        return train_loss / length, train_acc / length

    def test_speed(self):
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for data, target, path in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                # get data
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                prediction = output.data.max(1)[1]

        # get speed and time
        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("testing time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec)

        frame_rate = float(len(self.test_loader.dataset) / float(time_interval))
        print("FPS for this model is:%g" % frame_rate)
        return

    def test(self,*input, **kwargs):
        self.model.eval()
        correct = 0
        error_path = []
        ori_cat = []
        error_cat = []

        true_category_num = [0] * self.args.num_classes

        start_time = time.time()
        with torch.no_grad():
            for data, target, path in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                #get data
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                prediction = output.data.max(1)[1]

                results = prediction.eq(target.data)

                correct += prediction.eq(target.data).sum()

                # get error outputs
                for i in range(len(results.data)):
                    if results.data[i] == 0:
                        error_path.append(path[i])
                        ori_cat.append(target[i].item())
                        error_cat.append(prediction[i].item())
                # get recognitized number from each category
                for i in range(target.shape[0]):
                    if prediction[i] == target[i]:
                        true_category_num[target[i]] += 1

        acc = 100. * float(correct) / len(self.test_loader.dataset)

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("testing time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec)
        print("acc :", acc)

        with open(os.path.join(self.checkpoint_path, "error.txt"), "w") as f:
            f.write("%30s\t%2s\t%2s\n" % ("path", "ori_cat", "error_cat"))
            for i in range(len(error_path)):
                f.write("%30s\t%2d\t%2d\n" % (error_path[i], ori_cat[i], int(error_cat[i])))

        with open(os.path.join(self.checkpoint_path,"class_true_num_test.txt"), "w") as f:
            f.write("name\taccuracy\n")
            for i in range(len(true_category_num)):
                f.write("%d\t%d\n" % (i, true_category_num[i]))

        return

    def validation(self):
        self.model.eval()
        correct = 0
        top5_corret = 0

        with torch.no_grad():
            for datas in tqdm(self.val_loader, desc="evaluation", mininterval=1):
                if self.args.debug == True and correct >= 1: break
                data, target,path = datas
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                prediction = output.data.max(1)[1]
                correct += prediction.eq(target.data).sum()

                maxk = max((1, 5))
                target_resize = target.view(-1, 1)
                _, top5_pred = output.topk(maxk, 1, True, True)
                top5_corret += torch.eq(top5_pred, target_resize).sum().float().item()
        top5_acc = 100.0 * float(top5_corret) / len(self.val_loader.dataset)

        acc = 100. * float(correct) / len(self.val_loader.dataset)
        return acc, top5_acc



class CCML_Network(BaseModel):
    """
    Category-consistent deep network learning framework
    """
    def __init__(self, args,checkpoint_path):
        super().__init__()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:"+str(args.gpus) if use_cuda else "cpu")
        self.args = args
        self.checkpoint_path = checkpoint_path

        self.visual_path = os.path.join(self.checkpoint_path, "visual")
        makedirs(self.visual_path)

        # data loaders
        self.train_loader, self.test_loader, self.val_loader,self.test_speed_loader = self.load_datasets(self.args)
        # model
        self.model = self.getModel(modelName=args.modelName, num_classes=args.num_classes)
        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4, momentum=0.9)

        self.cls_loss = nn.CrossEntropyLoss().to(self.device)
        self.bce_loss = nn.BCELoss().cuda()

        self.CCML_loss_weight = self.args.ccml_loss_weight

        print(self.model)
        print(self.args)

    def getModel(self, modelName="CCML_VLF_net", num_classes=30):
        if modelName == "CCML_VLF_net":
            return CCML_VLF_net(num_classes=num_classes, small_inputs=True).to(self.device)

    def forward(self, data):
        data = data.to(self.device)
        output = self.model(data)
        return output

    def load_datasets(self, args):
        sizes_str = args.size
        sizes_ = sizes_str.split(",")
        siezs = [int(sizes_[i]) for i in range(len(sizes_))]

        crop_str = args.crop
        crops_ = crop_str.split(",")
        crops = [int(crops_[i]) for i in range(len(crops_))]

        args.train_dir = args.train_Root + "/train"
        args.val_dir = args.train_Root + "/val"
        args.test_dir = args.train_Root + "/test"

        if (not args.use_val == True):
            args.val_dir = args.test_dir

        train_loader, test_loader, val_loader, = load_data(args, crop_height=crops[0], crop_width=crops[1]
                                                          , height=siezs[0], width=siezs[1],
                                                          num_workers=args.num_workers,dataset_mode=args.dataset_mode)
        _, test_speed_loader, _ = load_data(args, crop_height=crops[0], crop_width=crops[1]
                                            , height=siezs[0], width=siezs[1],
                                            num_workers=args.num_workers, dataset_mode="vehicle_logo")
        return train_loader, test_loader, val_loader,test_speed_loader

    def train(self, epoch):
        self.model.train()
        step = 0
        train_loss = 0
        train_acc = 0

        count = 0
        for datas in tqdm(self.train_loader, desc="epoch " + str(epoch), mininterval=1):
            # data, mask, target = datas
            count += 1
            if self.args.debug == True and count == 100:break
            #get data
            inputs, mask, labels = datas
            inputs, mask, labels = inputs.to(self.device), mask.to(self.device), labels.to(self.device)

            self.adjust_learning_rate(self.optimizer, epoch, self.args)

            self.optimizer.zero_grad()
            output = self.forward(inputs)
            #classfication loss
            loss = self.cls_loss(output[0], labels)
            cls_loss_val = loss.item()

            # ccml loss
            ccml_loss = self.bce_loss(output[1], F.interpolate(mask, (output[1].size(2), output[1].size(3)),
                                                          mode="nearest")) * self.CCML_loss_weight
            ccml_loss_val = ccml_loss.item()
            loss += ccml_loss

            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            y_pred = output[0].data.max(1)[1]

            acc = float(y_pred.eq(labels.data).sum()) * 100.

            train_acc += acc
            step += 1
            if step % 100 == 0:
                print("[Epoch {0:4d}] cls Loss: {1:2.3f} ccml Loss: {2:2.3f} Acc: {3:.3f}%".format(epoch, cls_loss_val, ccml_loss_val, acc / len(labels)), end='')
                for param_group in self.optimizer.param_groups:
                    print(",  Current learning rate is: {}".format(param_group['lr']))


        length = len(self.train_loader.dataset)
        return train_loss / length, train_acc / length

    def visualization(self):
        self.model.eval()
        count = 0
        with torch.no_grad():
            for datas in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                count += 1
                if self.args.debug == True and count == 10: break
                # used for visualization and testing
                img, img_target, mask, img_path, mask_path = datas
                img, img_target = img.to(self.device), img_target.to(self.device)
                output = self.model(img)
                predicted_masks = output[1]
                # visualization
                visual_for_CCML(img, mask, predicted_masks, img_path, out_path=self.visual_path)
        return

    def calculate_IoU(self):
        self.model.eval()
        count = 0
        threshold_list = get_thresholds(bins=100, interval=(0, 1))
        acc_thres_iou_dic = {}
        for threshold in threshold_list:
            acc_thres_iou_dic[threshold] = [0, 0]

        with torch.no_grad():
            for datas in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                count += 1
                if self.args.debug == True and count == 10: break
                img, img_target, mask, img_path, mask_path = datas
                img, img_target = img.to(self.device), img_target.to(self.device)
                output = self.model(img)

                # caculate the IOU
                thre_IOU_dic = get_IOUs(pre_masks=output[1], gt_mask=mask, threshold_list=threshold_list)
                for key in thre_IOU_dic.keys():
                    acc_thres_iou_dic[key][0] += thre_IOU_dic[key][0]
                    acc_thres_iou_dic[key][1] += thre_IOU_dic[key][1]

        ###get IoU value
        threshold_str_ = "threshold:"
        iou_str_ = "iou:"
        for ii, key in enumerate(acc_thres_iou_dic.keys()):
            area_inter_acc = float(acc_thres_iou_dic[key][0]) / (acc_thres_iou_dic[key][1] + 1e-10)
            if ii == 0:
                threshold_str_ += "%g" % key
                iou_str_ += "%g" % area_inter_acc
            else:
                threshold_str_ += ",%g" % key
                iou_str_ += ",%g" % area_inter_acc
        threshold_str_ += "\n"
        iou_str_ += "\n"
        print(threshold_str_)
        print(iou_str_)

        with open(os.path.join(self.checkpoint_path,"iou.txt"), "w") as f:
            f.write(threshold_str_)
            f.write(iou_str_)
        return

    def test_speed(self):
        self.model.eval()
        count = 0
        start_time = time.time()
        with torch.no_grad():
            for datas in tqdm(self.test_speed_loader, desc="evaluation", mininterval=1):
                count += 1
                if self.args.debug == True and count >= 10: break
                # used for visualization and testing
                img, img_target, path = datas
                img, img_target = img.to(self.device), img_target.to(self.device)
                output = self.model(img,Test=True)
                prediction = output.data.max(1)[1]

        #get speed and time
        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("testing time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec)
        frame_rate = float(len(self.test_loader.dataset) / float(time_interval))
        print("FPS for this model is:%g" % frame_rate)

        return

    def test(self,Vis_files=False):
        self.model.eval()
        correct = 0
        error_path = []
        ture_path = []

        ori_cat = []
        error_cat = []
        true_category_num = [0] * self.args.num_classes

        count = 0
        start_time = time.time()
        with torch.no_grad():
            for datas in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                count += 1
                if self.args.debug == True and count == 10: break
                # used for visualization and testing
                img, img_target, mask, img_path, mask_path = datas
                img, img_target = img.to(self.device), img_target.to(self.device)
                output = self.model(img)
                prediction = output[0].data.max(1)[1]

                results = prediction.eq(img_target.data)
                correct += prediction.eq(img_target.data).sum()

                for i in range(len(results.data)):
                    if results.data[i] == 0:
                        error_path.append(img_path[i])
                        ori_cat.append(img_target[i].item())
                        error_cat.append(prediction[i].item())
                    else:
                        ture_path.append(img_path[i])

                # get recognitized number from each category
                for i in range(img_target.shape[0]):
                    if prediction[i] == img_target[i]:
                        true_category_num[img_target[i]] += 1

        acc = 100. * float(correct) / len(self.test_loader.dataset)

        #get speed and time
        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("testing time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec)
        print("acc :", acc)

        if Vis_files ==True:
            vis_files(ture_path, output_root=os.path.join(self.checkpoint_path,"true_files"))
            vis_files(error_path,output_root=os.path.join(self.checkpoint_path,"error_files"))

        with open(os.path.join(self.checkpoint_path,"error.txt"), "w") as f:
            f.write("%30s\t%2s\t%2s\n" % ("path", "ori_cat", "error_cat"))
            for i in range(len(error_path)):
                f.write("%30s\t%2d\t%2d\n" % (error_path[i], ori_cat[i], int(error_cat[i])))

        with open(os.path.join(self.checkpoint_path,"class_true_num_test.txt"), "w") as f:
            f.write("name\taccuracy\n")
            for i in range(len(true_category_num)):
                f.write("%d\t%d\n" % (i, true_category_num[i]))
        return


    def validation(self):
        self.model.eval()
        correct = 0
        top5_corret = 0

        with torch.no_grad():
            for datas in tqdm(self.val_loader, desc="evaluation", mininterval=1):
                if self.args.debug == True and correct >= 1: break
                data, target = datas
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                prediction = output[0].data.max(1)[1]
                correct += prediction.eq(target.data).sum()

                maxk = max((1, 5))
                target_resize = target.view(-1, 1)
                _, top5_pred = output[0].topk(maxk, 1, True, True)
                top5_corret += torch.eq(top5_pred, target_resize).sum().float().item()
        top5_acc = 100.0 * float(top5_corret) / len(self.val_loader.dataset)

        acc = 100. * float(correct) / len(self.val_loader.dataset)
        return acc, top5_acc