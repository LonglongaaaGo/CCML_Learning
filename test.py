
import argparse
from solver import trainer,tester
import time

def get_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--dataset_name', type=str, default="XMU", help='dataset name (default: XMU)')
    parser.add_argument('--framework', type=str, default="Classification_Network", help='dataset name (default: Classification_Network)')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size, (default: 128)')
    parser.add_argument('--dataset-mode', type=str, default="vehicle_logo", help='dataset, (default: CIFAR100)')

    parser.add_argument('--filename', type=str, default="vehicle_logo_1", help='dataset, (default: CIFAR100)')
    parser.add_argument('--debug', type=bool, default=False, help='use the val dataset(default: False)')
    parser.add_argument('--use_val', type=bool, default=False, help='use the val dataset(default: True)')


    parser.add_argument('--mask_path', type=str, default=None, help='mask  root for train')

    parser.add_argument('--train_Root', type=str, default="./data/vehicle_logo", help='dataset root for train')
    parser.add_argument('--train_dir', type=str, default="./data/vehicle_logo/car_trainData",
                        help='dataset root for train')
    parser.add_argument('--val_dir', type=str, default="./data/vehicle_logo/car_valData", help='dataset,root for val')
    parser.add_argument('--test_dir', type=str, default="./data/vehicle_logo/car_test", help='dataset, root for test')

    parser.add_argument('--num_classes', type=int, default=63, help='batch size, (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')

    parser.add_argument('--modelName', type=str, default="DenseNet", help='train_argu_dir for train student dir ')
    parser.add_argument('--size', type=str, default="64,64", help='size of the input images ')
    parser.add_argument('--crop', type=str, default="64,64", help='crop size of the input images ')

    parser.add_argument('--gpus', type=str, default="0", help=' gpu options, eg. 0, 1 ')
    parser.add_argument('--ccml_loss_weight', type=float, default=0.01, help='ccml_loss_weight, (default: 0.01)')


    args = parser.parse_args()

    return args

def test_HFUT_V1():
    args = get_args()
    args.debug = False
    args.framework = "Classification_Network"
    args.dataset_mode = "vehicle_logo"

    args.num_classes = 80
    args.train_Root = "/root/workspace/Data/HFUT-VL1_classify"
    args.modelName = "ReDenseNet_nv22"
    args.batch_size = 128
    args.size = "64,64"
    args.crop = "64,64"

    args.filename = "20210620" + "_" + args.modelName + "HFUT-VL1_" + args.size + "_" + args.crop + "_batch_" + str(
        args.batch_size)
    tester(args)




def test_HFUT_V2():

    args = get_args()
    args.debug = False
    args.framework = "Classification_Network"
    args.dataset_mode = "vehicle_logo"

    args.num_classes = 80
    args.train_Root = "/root/workspace/Data/HFUT-VL2_classify"
    args.modelName = "ReDenseNet_nv22"
    args.batch_size = 128
    #height, width
    args.size = "112,112"
    #height, width
    args.crop = "112,112"

    args.filename = "20210620" + "_" + args.modelName + "HFUT-VL2_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)
    tester(args)



def test_XMU():
    args = get_args()
    args.framework = "Classification_Network"
    args.dataset_mode = "vehicle_logo"
    args.debug = False
    args.num_classes = 10
    args.train_Root = "/root/workspace/Data/XMU_data_7_3_split"
    args.modelName = "ReDenseNet_nv22"
    args.size = "70,70"
    args.crop = "70,70"
    args.batch_size = 128
    args.filename = "20210620" + "_" + args.modelName + "XMU_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)
    tester(args)


def test_CompCar():

    args = get_args()
    args.debug = False
    args.framework = "Classification_Network"
    args.dataset_mode = "vehicle_logo"

    args.num_classes = 68
    args.train_Root = "/root/workspace/Data/sv_data_by_logo_0811/"
    args.modelName = "ReDenseNet_nv22"
    args.batch_size = 32
    args.size = "256,256"
    args.crop = "224,224"

    args.filename = "20210620"+ "_" + args.modelName + "_sv_data_by_logo_" + args.size + "_" + args.crop + "_batch_" + str(
        args.batch_size) + "_bce"
    tester(args)



def test_VLD_45B():
    args = get_args()
    args.framework = "Classification_Network"
    args.dataset_mode = "vehicle_logo"

    args.train_Root = "/root/workspace/Workspace/Data/VLD-45-B_class_30000"
    args.debug = False
    args.num_classes = 45
    args.modelName = "ReDenseNet_nv22"
    args.batch_size = 8
    args.size = "512,512"
    args.crop = "512,512"
    args.num_workers = 8

    args.filename = "20210620" + "_" + args.modelName + "VLD_45B_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)
    tester(args)

    return


def ccml_test_HFUT_V1():

    args = get_args()
    args.framework = "CCML_Network"
    args.dataset_mode = "CCML_vehicle_logo"

    args.num_classes = 80
    args.train_Root = "/root/workspace/Data/HFUT-VL1_classify"
    args.modelName = "pm_ReDenseNet_nv22"
    args.batch_size = 128
    args.size = "64,64"
    args.crop = "64,64"
    args.debug = True

    args.mask_path = "/root/workspace/Data/HFUT-VL1_classify/all_mask"

    args.filename = "20210620" + "_" + args.modelName + "HFUT-VL1_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)+ "_bce_" + str(args.ccml_loss_weight)
    tester(args)


def ccml_test_HFUT_V2():

    args = get_args()
    args.framework = "CCML_Network"
    args.dataset_mode = "CCML_vehicle_logo"
    args.num_classes = 80
    args.train_Root = "/root/workspace/Data/HFUT-VL2_classify"
    args.modelName = "pm_ReDenseNet_nv22"
    args.batch_size = 128
    args.size = "112,112"
    args.crop = "112,112"
    args.debug = True

    args.mask_path = "/root/workspace/Data/HFUT-VL2_classify/all_mask"

    args.filename = "20210620" + "_" + args.modelName + "HFUT-VL2_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)+ "_bce_" + str(args.ccml_loss_weight)

    tester(args)



def ccml_test_VLD_45B():

    args = get_args()
    args.framework = "CCML_Network"
    args.dataset_mode = "CCML_vehicle_logo"

    args.num_classes = 45
    # args.filename = "pmdcl_FLD_HFUT-VL1_64_64_batch_128_bce_2"
    args.modelName = "pm_ReDenseNet_nv22"
    args.batch_size = 8

    args.num_workers = 8

    args.size = "512,512"
    args.crop = "512,512"
    args.debug = False
    args.train_Root = "/root/workspace/Workspace/Data/VLD-45-B_class_30000"
    args.mask_path = "/root/workspace/Workspace/Data/VLD-45-B_class_30000/512_all_mask"

    args.filename = "20210620" + "_" + args.modelName + "VLD_45B_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)+ "_bce_" + str(args.ccml_loss_weight)

    tester(args)

    return



def ccml_test_XMU():

    args = get_args()
    args.framework = "CCML_Network"
    args.dataset_mode = "CCML_vehicle_logo"

    args.num_classes = 10
    args.train_Root = "/root/workspace/Data/XMU_data_7_3_split"
    args.modelName = "pm_ReDenseNet_nv22"
    args.batch_size = 128
    args.size = "70,70"
    args.crop = "70,70"
    args.debug = False

    args.mask_path = "/root/workspace/Data/XMU_data_7_3_split/all_mask"

    args.filename = "20210620" + "_" + args.modelName + "_XMU_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)+ "_bce_" + str(args.ccml_loss_weight)
    tester(args)


def ccml_test_CompCar():

    args = get_args()
    #frame work name
    args.framework = "CCML_Network"
    #
    args.dataset_mode = "CCML_vehicle_logo"

    args.num_classes = 68
    #data root for training
    args.train_Root = "/root/workspace/Workspace/Data/sv_data_by_logo_0811/"
    args.modelName = "pm_ReDenseNet_nv22"
    args.batch_size =32
    args.size = "256,256"
    args.crop = "224,224"
    args.debug = True
    #root for masks
    args.mask_path = "/root/workspace/Workspace/Data/sv_data_by_logo_0811/all_mask"

    args.filename = "20210620" + "_" + args.modelName + "_sv_data_by_logo_" + args.size + "_" + args.crop + "_batch_" + str(args.batch_size)+ "_bce_" + str(args.ccml_loss_weight)

    tester(args)


def main():
    args = get_args()

    if args.dataset_name == "XMU":
        if args.framework == "Classification_Network":
            test_XMU()
        elif args.framework == "CCML_Network":
            ccml_test_XMU()

    elif args.dataset_name == "HFUT_VL1":
        if args.framework == "Classification_Network":
            test_HFUT_V1()
        elif args.framework == "CCML_Network":
            ccml_test_HFUT_V1()

    elif args.dataset_name == "HFUT_VL2":
        if args.framework == "Classification_Network":
            test_HFUT_V2()
        elif args.framework == "CCML_Network":
            ccml_test_HFUT_V2()

    elif args.dataset_name == "CompCars":
        if args.framework == "Classification_Network":
            test_CompCar()
        elif args.framework == "CCML_Network":
            ccml_test_CompCar()

    elif args.dataset_name == "VLD-45":
        if args.framework == "Classification_Network":
            test_VLD_45B()
        elif args.framework == "CCML_Network":
            ccml_test_VLD_45B()

if __name__ == '__main__':
    main()

    # test_XMU()
    # test_HFUT_V1()
    # test_HFUT_V2()
    # test_VLD_45B()
    # test_CompCar()
    # ccml_test_XMU()
    # ccml_test_HFUT_V1()
    # ccml_test_HFUT_V2()
    # ccml_test_VLD_45B()
    # ccml_test_CompCar()


