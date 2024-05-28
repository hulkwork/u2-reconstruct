import logging
import os
import random
from typing import Dict, List

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

def rotate_pil_imgage(pil_img:Image.Image, rotations : List[int] = None) -> List[Image.Image]:
    if rotations is None:
        return [pil_img]
    return [pil_img.rotate(angle) for angle in rotations]


def mask_image(img, n_x: int, n_y: int, x: int = None, y: int = None):

    width, height = img.size

    square_size_x = width // n_x
    square_size_y = height // n_y

    if x is None:
        x = random.randint(0, width - square_size_x)
    if y is None:
        y = random.randint(0, height - square_size_y)

    masked_img = img.copy()
    draw = ImageDraw.Draw(masked_img)
    draw.rectangle([x, y, x + square_size_x, y + square_size_y], fill="black")

    return masked_img

def get_subgrid_coords(n, m, p, q):
    coords = []
    for i in range(0, n, p):
        for j in range(0, m, q):
            top_left = (i, j)
            coords.append(top_left)
    return coords

class FilesDataset(Dataset):
    def __init__(
        self,
        imgs_files: List[str],
        resize: int = 256,
        normalized: bool = True,
        metadata: Dict[str, str] = None,
        is_transfo :bool = False
    ):
        self.resize = resize
        self.normalize = normalized
        self.ids = []
        logging.info(f"Creating dataset with {len(self.ids)} examples")
        self.metadata = metadata
        self.masker = get_subgrid_coords(n=resize, m=resize, p = resize // 2, q=resize //2)
        self.transfo = []
        self.is_transfo = is_transfo
        if is_transfo:
            for item in imgs_files:
                for maske in self.masker:
                    self.transfo.append(maske)
                    self.ids.append(item)
        else:
            self.ids = imgs_files


    def transformations(self, i:int):
        img_file = self.ids[i]
        x,y = self.transfo[i]
        img = Image.open(img_file)
        img = img.resize((self.resize, self.resize))
        return mask_image(img, n_x=2, n_y=2, x=x,y=y)

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
        result = {}
        img_file = self.ids[i]

        assert os.path.exists(img_file), f"Either no image found for {img_file}"
        img = Image.open(img_file)
        if self.normalize:
            std = 1.0
        else:
            std = 225.0
        img = self.preprocess(img, self.resize, normalize=self.normalize)
        img_noise = AddGaussianNoise(mean=0.0, std=std)(torch.from_numpy(img))

        if self.is_transfo:
            img_masked = self.transformations(i)
            img_masked = self.preprocess(img_masked, self.resize, normalize=self.normalize)
            result["noised"] =  img_noise + torch.from_numpy(img_masked)
            result["masked"] = torch.from_numpy(img_masked)

        result.update(
            {"image": torch.from_numpy(img),
            "output": torch.from_numpy(img),
            "noise": img_noise,
            "file_path": img_file,
            "id": i,
            }
        )
        if self.metadata is not None:
            result.update(self.metadata)
        return result
