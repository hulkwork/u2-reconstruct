from os import listdir

import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class ReconstructDataset(Dataset):
    def __init__(self, imgs_dir : str, resize : int=256, normalized : bool = True):
        self.imgs_dir = imgs_dir
        self.resize = resize
        self.normalize = normalized
        
        self.ids = list(glob(imgs_dir))
                    
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, new_size, normalize : bool = True):
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

        assert os.path.exists(img_file), \
            f'Either no image found for {img_file}'
        img = Image.open(img_file)
        img = self.preprocess(img, self.resize, normalize=self.normalize)
        
        return {'image': torch.from_numpy(img), 'output': torch.from_numpy(img)}