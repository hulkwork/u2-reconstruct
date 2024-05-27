import logging
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def mask_image(img, n: int, x: int = None, y: int = None):

    width, height = img.size

    square_size = min(width, height) // n

    if x is None:
        x = random.randint(0, width - square_size)
    if y is None:
        y = random.randint(0, height - square_size)

    masked_img = img.copy()
    draw = ImageDraw.Draw(masked_img)
    draw.rectangle([x, y, x + square_size, y + square_size], fill="black")

    return masked_img


class FilesDataset(Dataset):
    def __init__(
        self, imgs_files: List[str], resize: int = 256, normalized: bool = True
    ):
        self.resize = resize
        self.normalize = normalized
        self.ids = imgs_files
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, new_size, normalize: bool = True):
        newW, newH = new_size, new_size
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2, 0, 1))
        if normalize:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.ids[i]

        assert os.path.exists(img_file), f"Either no image found for {img_file}"
        img = Image.open(img_file)
        img_masked = mask_image(img, n=2)

        img = self.preprocess(img, self.resize, normalize=self.normalize)
        img_masked = self.preprocess(img_masked, self.resize, normalize=self.normalize)
        if self.normalize:
            std = 1.0
        else:
            std = 225.0
        img_noise = AddGaussianNoise(mean=0.0, std=std)(torch.from_numpy(img))
        return {
            "image": torch.from_numpy(img),
            "masked": torch.from_numpy(img_masked),
            "output": torch.from_numpy(img),
            "noise": img_noise,
            "noised": img_noise + torch.from_numpy(img_masked),
        }
