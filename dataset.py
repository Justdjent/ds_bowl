import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import prepare_data
import os

data_path = Path('data')


class NucleiDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # print(self.file_names)
        # print(idx)
        # print(self.file_names[idx], len(self.file_names), idx)
        img_file_name = self.file_names[idx]
        img = load_image(img_file_name, self.mode)
        if self.mode == 'train':
            mask = load_mask(img_file_name, self.mode)
        else:
            mask = None
        img, mask = self.transform(img)

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return to_float_tensor(img),\
                       torch.from_numpy(np.expand_dims(mask, 0)).float()
                       # torch.from_numpy(np.expand_dims(seed, 0)).float(),\
                       # torch.from_numpy(np.expand_dims(border, 0)).float()
            else:
                # return to_float_tensor(img), torch.from_numpy(mask).long()
                return to_float_tensor(img), to_float_tensor(mask)
        else:
            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path, mode):
    path_ = "data/stage1_train_/{}/images/{}.png".format(path, path)
    if mode != 'train':
        path_ = "data/cropped_test/{}".format(path)
    if not os.path.isfile(path_):
        print('{} was empty'.format(path_))
    img = cv2.imread(str(path_))

    # print(path_, img.shape)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, mode):
    path_ = "data/stage1_train_/{}/masksmask.png".format(path)
    if mode != 'train':
        path_ = "data/stage1_test/{}/images/{}.png".format(path, path)
    if not os.path.isfile(path_):
        print('{} was empty'.format(path_))
    factor = prepare_data.binary_factor
    mask = cv2.imread(str(path_))
    kernel = np.ones((4, 4), np.uint8)
    seed = cv2.erode(mask[:, :, 0], kernel, iterations=1)
    border = mask[:, :, 0] - seed
    mask[:, :, 1] = np.zeros(seed.shape)
    mask[:, :, 1] = seed
    mask[:, :, 2] = np.zeros(seed.shape)
    mask[:, :, 2] = border

    return (mask / factor).astype(np.uint8)
