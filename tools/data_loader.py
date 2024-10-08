import os
import random
import numpy as np

from torch.utils import data
from torchvision import transforms as T

def load_data_Charles(image_path, config):
    data_in = np.fromfile(image_path, dtype=np.float64)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)
    input_sets = data_pairs[:, :, :config.INPUT_C]
    label_sets = data_pairs[:, :, config.INPUT_C:config.INPUT_C+config.LABEL_C]
    label_sets = label_sets

    return input_sets,label_sets

def load_data_Charles_test(image_path, config):
    data_in = np.fromfile(image_path, dtype=np.float64)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)
    input_sets = data_pairs[:, :, :config.INPUT_C]
    label_sets = data_pairs[:, :, config.INPUT_C:config.INPUT_C + config.LABEL_C]
    label_sets = label_sets

    return input_sets,label_sets


class ImageFolder(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, crop_key, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))
        self.image_paths.sort(reverse=True)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        if self.mode == 'brain':
            image, GT = load_data_Charles_test(image_path, self.config)
        else:
            image,GT = load_data_Charles(image_path, self.config)

        if self.crop_key:
            # -----RandomCrop----- #
            (w, h, c) = image.shape
            th, tw = self.crop_size, self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - th)
            if w <= th and h <= th:
                print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                return
            image = image[i:i + th, j:j + th,:]
            GT = GT[i:i + th, j:j + th,:]

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

class ImageFolder_reg(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, crop_key, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        if self.mode == 'brain':
            image, GT = load_data_Charles_test(image_path, self.config)
        else:
            image,GT = load_data_Charles(image_path, self.config)

        if self.crop_key:
            # -----RandomCrop----- #
            (w, h, c) = image.shape
            th, tw = self.crop_size, self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - th)
            if w <= th and h <= th:
                print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                return
            image = image[i:i + th, j:j + th,:]

        vx = GT[:,:,3].max()
        vz = GT[:,:,4].max()
        rot = GT[:,:,5].max()

        GT = np.array([vx,vz,rot])
        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        #GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, config, crop_key, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, config=config, crop_key=crop_key, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader

def get_loader_reg(image_path, config, crop_key, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder_reg(root=image_path, config=config, crop_key=crop_key, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader