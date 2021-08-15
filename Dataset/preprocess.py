import torch

from torchvision import transforms
from Dataset import data_folder as fd


def load_data(args,crop_height =112, crop_width = 112,height = 112,width = 112,num_workers=4,dataset_mode="vehicle_logo"):
    train_loader = None
    test_loader = None
    val_loader = None

    if dataset_mode == "vehicle_logo":
        # Data transforms
        mean = [0.5071, 0.4867, 0.4408]
        stdv = [0.2675, 0.2565, 0.2761]
        transform_train = transforms.Compose([
            transforms.RandomCrop((crop_height,crop_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        #center cropping will be disabled in the testing phase
        transform_test = transforms.Compose([
            # transforms.CenterCrop((crop_height, crop_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        print(args.train_dir)
        trainDataset = fd.ImageFolder(root = args.train_dir, transform = transform_train,height = height,width = width)
        train_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        testDataset = fd.ImageFolder(root=args.test_dir, transform=transform_test,height = height,width = width)
        test_loader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        valDataset = fd.ImageFolder(root=args.val_dir, transform=transform_test,height = height,width = width)
        val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    elif args.dataset_mode == "CCML_vehicle_logo":
        # Data transforms
        mean = [0.5071, 0.4867, 0.4408]
        stdv = [0.2675, 0.2565, 0.2761]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        #center cropping will be disabled in the testing phase
        transform_test = transforms.Compose([
            # transforms.CenterCrop((crop_height, crop_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])

        mask_tran = transforms.Compose([
            transforms.ToTensor(),
        ])

        print(args.train_dir)
        trainDataset = fd.CCML_Train_ImageFolder(root = args.train_dir,mask_path =args.mask_path,
                                             train_type=True, transform = transform_train,mask_transform=mask_tran,
                                              crop_height = crop_height,crop_width=crop_width,height = height ,width = width )
        train_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        #center cropping will be disabled in the testing phase
        testDataset = fd.CCML_Test_ImageFolder(root=args.test_dir, mask_path=args.mask_path, transform=transform_test, mask_transform=mask_tran,
                                                 crop_height=height, crop_width=width, height=height,width=width)

        test_loader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        #center cropping will be disabled in the validation phase
        valDataset = fd.CCML_Train_ImageFolder(root=args.val_dir,mask_path =args.mask_path,
                                           train_type=False, transform=transform_test,
                                               crop_height = height,crop_width=width,height = height ,width =width )
        val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, test_loader,val_loader
