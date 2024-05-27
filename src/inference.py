import argparse
import logging
import sys
import os

import torch
from tqdm import tqdm
import sys 
import cv2
from torch.utils.data import DataLoader
from torch.nn import MSELoss

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '../'))

from src.model import UNet
from src.losses.losses import SSIM_Loss, CustomLoss
from src.eval import scale_to_image
from src.dataset import FilesDataset


dir_checkpoint = 'checkpoints/'

def infer(net:UNet, loader : DataLoader, 
                   device,
                   batch_size : int=1,
                    normalized : bool = False, 
                    log_dir : str = "./"):
    total_params = sum(param.numel() for param in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_train = len(loader)
    


    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Training size:   {n_train}
        Device:          {device.type}
        Nb Parameters:   {total_params}
        N. Train Params: {trainable_params}
        
    ''')

    criterion1 = SSIM_Loss(data_range=1.0, channel=3, size_average=False, nonnegative_ssim=True)
    criterion = CustomLoss(losses=[criterion1, MSELoss()], weights=[0.9, 0.01])
    global_step = 0
    net.eval()

    with tqdm(total=n_train, desc='Inference', unit='img') as pbar:
        i = 0
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['output']
            if imgs.shape[1] != net.n_channels:
                logging.info(imgs.shape[1])
                continue

            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32
            true_masks = true_masks.to(device=device, dtype=mask_type)
            with torch.no_grad():
                mask_pred = net(imgs)
                current_ssim = criterion(mask_pred, true_masks)
                logging.info(str(i)+" criterion" + str(current_ssim))
                

            gt_image = scale_to_image(imgs, normalize=normalized)
            pred = scale_to_image(mask_pred, normalize=normalized)
            diff_image = diff_mask(gt_image, pred)
            im_v = cv2.hconcat([gt_image, pred, diff_image])
            cv2.imwrite(os.path.join(log_dir, str(i) + ".png"), im_v)
            i += 1
              
            pbar.update(imgs.shape[0])
           
        

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    
    parser.add_argument('-i', '--item', dest='item', type=str, default=None,
                        help='Item to train')

    return parser.parse_args()

def diff_mask(image1, image2):
    difference = cv2.subtract(image1, image2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 100, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    return difference

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.load_state_dict(
            torch.load(args.load, map_location=device)
    )
    logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    batch_size = 1
    """
    bottle/
    cable/
    capsule/
    carpet/
    grid/
    hazelnut/
    leather/
    metal_nut/
    pill/
    screw/
    tile/
    toothbrush/
    transistor/
    wood/
    zipper/
    """
    p1 = "/home/michou/mvtec/data/bottle/test/broken_large/000.png"
    p2 = "/home/michou/mvtec/data/pill/test/scratch/000.png"
    dataset = FilesDataset(imgs_files=[p1, p2], resize=256, normalized=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    os.makedirs('output/', exist_ok=True)
    infer(net=net, loader=loader, device=device, normalized=False, log_dir='output/', batch_size=batch_size)
    # python src/inference.py --load /home/michou/mvtec/u2-reconstruct/runs/bottle/2024-05-25T14:39:28.519102/model_epoch14.pth