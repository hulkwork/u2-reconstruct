import os 
import torch
from tqdm import tqdm
import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def scale_to_image(img, normalize : bool = True):
    if normalize:
        img = (img * 255.0)
    img = img.squeeze().detach().cpu().numpy().astype(np.uint8)
    img_trans = img.transpose((1,2,0))
    return img_trans

def eval_net(net, loader, device,criterion : torch.nn.Module, epoch : int, dir_ : str, normalize = False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 
    n_val = len(loader)  # the number of batch
    tot = 0
    path_result = os.path.join(dir_)
    os.makedirs(path_result, exist_ok=True)


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        i = 0
        for batch in loader:
            imgs, true_masks = batch['image'], batch['output']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                current_ssim = criterion(mask_pred, true_masks)
                tot += current_ssim

            gt_image = scale_to_image(imgs, normalize=normalize)
            pred = scale_to_image(mask_pred, normalize=normalize)
            im_v = cv2.hconcat([gt_image, pred])
            cv2.imwrite(os.path.join(dir_, str(epoch)+"_"+ str(i) + ".png"), im_v)
            i += 1
            
            
            pbar.update()
    logging.info(f'Total ssim {tot / n_val}')
    return tot / n_val