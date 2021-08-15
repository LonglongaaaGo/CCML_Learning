import torch
from Models.VLF_net import pm_ReDenseNet_nv22,ReDenseNet_nv22


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def getModel(modelName = "DenseNet",num_classes = 30):

    if modelName == "pm_ReDenseNet_nv22":
        return pm_ReDenseNet_nv22(num_classes=num_classes, small_inputs=True,use_dcl = False).to(device)

    elif modelName == "ReDenseNet_nv22":
        return ReDenseNet_nv22(num_classes=num_classes,small_inputs=True).to(device)




def torch_summary():
    # from torchsummary import summary
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    train_list = ["pm_ReDenseNet_nv22"]


    for i in range(len(train_list)):
        print("="*100)
        print(train_list[i])
        model = getModel(train_list[i],num_classes=45).to(device)

        summary(model, input_size=(1, 3, 512, 512),depth = 5)



if __name__ == '__main__':

    torch_summary()