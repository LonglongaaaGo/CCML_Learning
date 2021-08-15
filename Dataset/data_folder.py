import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random
import torch
from PIL import ImageStat



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    """
    :param dir: a root dir contains subdirectories
    :return: the list for category name and index
    """
    classes = [d for d in os.listdir(dir)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def getAllFile(file_list,dir,class_to_idx,extensions,target):
    """
    :param file_list: 递归保存的容器
    :param dir: 递归的子目录
    :param class_to_idx: 类别名称到id的映射
    :param extensions: 检查是否符合某种数据类型
    :param target: target表示当前的类别名字
    :return: 无返回
    """

    if not os.path.isdir(dir):
        return

    for root, dirs, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                file_list.append(item)
        for dname in sorted(dirs):
            new_dir = os.path.join(root, dname)

            getAllFile(file_list, new_dir, class_to_idx, extensions, target)


def make_dataset(dir, class_to_idx, extensions):
    images = []

    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, dirs, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images




class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        height (int): height for resized image
        width (int): width for resized image
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        images (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, height = 112,width = 112):
        classes, class_to_idx = find_classes(root)
        images= make_dataset(root, class_to_idx, extensions)
        if len(images) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = images

        self.transform = transform
        self.target_transform = target_transform
        self.height = height
        self.width = width

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target,path) where target is class_index of the target class.
        """
        img_path, img_target = self.images[index]

        img = Image.open(img_path).convert('RGB')

        # sample = self.loader(path)

        # rand_flip = random.uniform(0.0,1.0)
        # if(rand_flip>0.5):
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # rand_rotate = random.uniform(0.0,360.0)
        # img = img.rotate(int(rand_rotate))

        #resize(w,h) for the PIL
        img = img.resize((int(self.width),int(self.height)), Image.ANTIALIAS)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            img_target = self.target_transform(img_target)

        return img,img_target,img_path


    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class CCML_Test_DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
    root/
        test/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        train/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        mask/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png

    Args:
        root (string): Root directory path.
        mask_path  (string): Root directory path of the category-consistent masks.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        mask_transform (callable, optional): A function/transform that takes in the
            masks and transforms it.
        loader (callable, optional): A function to load an image given its path.
        crop_height: height for cropped image
        crop_width: width for cropped image
        height: height for resized image
        width: width for resized image

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        images (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions,mask_path, transform=None, mask_transform=None,
                 crop_height=112,crop_width=112,height = 112,width = 112):

        classes, class_to_idx = find_classes(root)
        images= make_dataset(root, class_to_idx, extensions)

        if len(images) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = images

        self.transform = transform
        self.mask_transform = mask_transform
        self.height = height
        self.width = width
        self.size_ = [height, width]
        self.crop_ = [crop_height,crop_width]
        self.mask_path = mask_path
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, img_target, mask, img_path, mask_path) 
            where img_target is class_index of the target class.
        """
        img_path, img_target = self.images[index]

        img = Image.open(img_path).convert('RGB')

        temps = img_path.split("/")
        mask_path = os.path.join(self.mask_path,temps[-2]+"/"+temps[-1])
        # mask_path = mask_path.replace("_$add$_augm", "")

        mask = Image.open(mask_path).convert('1')

        img = img.resize((int(self.width),int(self.height)), Image.ANTIALIAS)

        if self.transform is not None: img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, img_target, mask, img_path, mask_path

    def __len__(self):
        return len(self.images)


class CCML_Train_DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
    root/
        test/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        train/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        mask/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
    Args:
        root (string): Root directory path.
        mask_path (string): Root directory path of the category-consistent masks.
        train_type (bool): if it is in the training
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        mask_transform (callable, optional): A function/transform that takes in the
            masks and transforms it.
        loader (callable, optional): A function to load an image given its path.
        crop_height (int): height for cropped image
        crop_width (int): width for cropped image
        height (int): height for resized image
        width (int): width for resized image

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions,mask_path,train_type = True, transform=None, mask_transform=None, crop_height=112,crop_width=112,height = 112,width = 112):

        classes, class_to_idx = find_classes(root)
        images= make_dataset(root, class_to_idx, extensions)

        if len(images) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = images

        self.transform = transform
        self.mask_transform = mask_transform
        self.height = height
        self.width = width
        self.size_ = [height, width]
        self.crop_ = [crop_height,crop_width]
        self.traintype = train_type
        self.mask_path = mask_path
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            if self.traintype == True:
                return tuple: (img,mask,img_target) where img_target is class_index of the target class.
                
            elif self.traintype == False:
                return tuple: (img,img_target) where img_target is class_index of the target class.
        """
        img_path, img_target = self.images[index]

        img = Image.open(img_path).convert('RGB')

        if not self.traintype:
            img = img.resize((int(self.width),int(self.height)), Image.ANTIALIAS)

            if self.transform is not None: img = self.transform(img)
            return img,img_target

        temps = img_path.split("/")
        mask_path = os.path.join(self.mask_path,temps[-2]+"/"+temps[-1])

        mask_path = mask_path.replace("_$add$_augm", "")

        mask = Image.open(mask_path).convert('1')
        # mask = mask.resize((int(self.height), int(self.width)), Image.ANTIALIAS)
        img,mask = self.data_argu(img, mask, crop_size=(self.crop_[0], self.crop_[1]), resize_size=(self.size_[0],self.size_[1]))

        if self.transform is not None:
            img = self.transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img,mask,img_target

    def data_argu(self, img, mask, crop_size=(224, 224), resize_size=(256, 256)):
        #resize_size[1],resize_size[0] == > w,h
        #resize(w,h)
        img = img.resize((resize_size[1],resize_size[0]), Image.ANTIALIAS)
        mask = mask.resize((resize_size[1], resize_size[0]), Image.NEAREST)
        w, h = img.size

        # degree = random.randint(-15, 15)
        # img = img.rotate(degree)
        # mask = mask.rotate(degree)

        #crop_size[1],crop_size[0] ==> w,h
        rw = random.randint(0, w - crop_size[1])
        rh = random.randint(0, h - crop_size[0])
        img = img.crop((rw, rh, int(rw + crop_size[1]), int(rh + crop_size[0])))
        mask = mask.crop((rw, rh, int(rw + crop_size[1]), int(rh + crop_size[0])))

        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif','gif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



def Z_score(img):
    """
    zxyd = Ixyd - μd /σd ,
     μd = 1 /W*H *sum (Ixyd)
     σd = 1 /W*H *sum(Ixyd - μd )^2
    :param img: PIL image
    :return: img  c,h,w
    """
    img = np.asarray(img).astype(np.float32)

    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=-1)

    mean = np.mean(img[img[..., 0] > 40.0], axis=0)
    std = np.std(img[img[..., 0] > 40.0], axis=0)
    # assert (len(mean) == 3 and len(std) == 3) or (len(mean) == 1 and len(std) == 1)
    img = (img - mean) / std
    #h,w,c => c,h,w
    img = np.transpose(img, (2, 0, 1))
    # # img = img/255

    return img,mean,std

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        height (int): height for resized image
        width (int): width for resized image
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, height = 112,width = 112):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform, height = height,width = width)
        self.imgs = self.images





class CCML_Train_ImageFolder(CCML_Train_DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
    root/
        test/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        train/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        mask/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png

    Args:
        root (string): Root directory path.
        mask_path  (string): Root directory path of the category-consistent masks.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        mask_transform (callable, optional): A function/transform that takes in the
            masks and transforms it.
        loader (callable, optional): A function to load an image given its path.
        crop_height (int) : height for cropped image
        crop_width (int): width for cropped image
        height (int): height for resized image
        width (int): width for resized image

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, mask_path ,train_type = True,transform=None, mask_transform=None,
                 loader=default_loader, crop_height = 112,crop_width=112,height = 112,width = 112):
        super(CCML_Train_ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,mask_path =mask_path,train_type = train_type,
                                          transform=transform, mask_transform=mask_transform,
                                                 crop_height = crop_height,crop_width=crop_width,height = height,width = width)
        self.imgs = self.images





class CCML_Test_ImageFolder(CCML_Test_DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
    root/
        test/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        train/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        mask/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png

    Args:
        root (string): Root directory path.
        mask_path  (string): Root directory path of the category-consistent masks.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        mask_transform (callable, optional): A function/transform that takes in the
            masks and transforms it.
        loader (callable, optional): A function to load an image given its path.
        crop_height: height for cropped image
        crop_width: width for cropped image
        height: height for resized image
        width: width for resized image

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, mask_path ,transform=None, mask_transform=None,
                 loader=default_loader,crop_height = 112,crop_width=112,height = 112,width = 112):

        super(CCML_Test_ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,mask_path =mask_path,
                                          transform=transform,mask_transform=mask_transform,
                                        crop_height = crop_height,crop_width=crop_width,height = height,width = width)
        self.imgs = self.images
