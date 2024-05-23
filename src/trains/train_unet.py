import argparse
import logging
import sys
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import sys 
sys.path.insert(0, "/home/michou/mvtec/u2-reconstruct")

from src.model import UNet
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss
from src.dataset import ReconstructDataset
from src.losses import SSIM_Loss, CustomLoss
from src.eval import eval_net
from torch.utils.data import DataLoader, random_split


dir_checkpoint = 'checkpoints/'

def train_approach(net:UNet, train_loader : DataLoader, 
                   validation_loader : DataLoader, 
                   writer : SummaryWriter,
                   device,
                   epochs :int =10,
                   batch_size : int=1,
                    lr : float=0.001, 
                    normalized : bool = False, 
                    log_dir : str = "./"):
    total_params = sum(param.numel() for param in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_train = len(train_loader)
    n_val = len(validation_loader)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Nb Parameters:   {total_params}
        N. Train Params: {trainable_params}
        
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion1 = SSIM_Loss(data_range=1.0, channel=3, size_average=False, nonnegative_ssim=True)
    criterion = CustomLoss(losses=[criterion1, MSELoss()], weights=[0.9, 0.01])
    global_step = 0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['output']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
        global_step += 1        
        val_score = eval_net(net, validation_loader, device,criterion=criterion, epoch = epoch, 
                             dir_ =log_dir +"/" + str(epoch), normalize=normalized)
        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Validation SSIM loss: {}'.format(val_score))
        writer.add_scalar('Loss/test', val_score, global_step)
        writer.add_images('images', imgs, global_step)
        
        path_checkpoint = os.path.join(log_dir,f'model_epoch{epoch + 1}.pth' )
        torch.save(net.state_dict(),path_checkpoint)
        logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


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

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    batch_size = 1
    val_percent = 0.1
    dataset = ReconstructDataset(imgs_dir="/home/michou/mvtec/data/pill/train/good/*.png", resize=256, normalized=False)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    log_dir = f"runs/" + datetime.now().isoformat()
    writer = SummaryWriter(log_dir=log_dir)
    
    
    train_approach(net=net,
                   train_loader=train_loader, 
                   validation_loader=val_loader,
                   epochs=args.epochs,
                   batch_size=args.batchsize,
                   writer=writer, 
                   lr=args.lr,
                   device=device,
                   normalized=False,
                   log_dir=log_dir
                   )
