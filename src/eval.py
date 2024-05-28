import logging
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.utils import diff_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def scale_to_image(img, normalize: bool = True):
    if normalize:
        img = img * 255.0
    img = img.squeeze().detach().cpu().numpy().astype(np.uint8)
    img_trans = img.transpose((1, 2, 0))
    return img_trans


def eval_net(
    net,
    loader,
    device,
    criterion: torch.nn.Module,
    epoch: int,
    dir_: str,
    normalize=False,
):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    path_result = os.path.join(dir_)
    os.makedirs(path_result, exist_ok=True)

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        i = 0
        for batch in loader:
            imgs, true_masks, noisy = batch["image"], batch["output"], batch["noise"]
            masked = batch["masked"]
            if imgs.shape[1] != net.n_channels:
                logging.info(imgs.shape[1])
                continue
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                current_ssim = criterion(mask_pred, true_masks)
                tot += current_ssim

            gt_image = scale_to_image(imgs, normalize=normalize)
            pred = scale_to_image(mask_pred, normalize=normalize)
            noisies = scale_to_image(noisy, normalize=normalize)
            masked = scale_to_image(masked, normalize=normalize)
            dif_pred_gt = diff_image(gt_image, pred)
            images = [gt_image, pred, dif_pred_gt, masked, noisies]
            if "gt" in batch:
                gt_img_broke = batch["gt"]
                gt_img_broke = gt_img_broke.expand(3, *gt_img_broke.shape[1:])
                gt_img_broke = scale_to_image(gt_img_broke, normalize=normalize)
                images.append(gt_img_broke)
                images.append(diff_image(dif_pred_gt, gt_img_broke))

            im_v = cv2.hconcat(images)
            cv2.imwrite(os.path.join(dir_, str(epoch) + "_" + str(i) + ".png"), im_v)
            i += 1

            pbar.update()
    logging.info(f"Total ssim {tot / n_val}")
    return tot / n_val
