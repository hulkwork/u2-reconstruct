import argparse
import logging
import os
from typing import Dict

########## Torch packages ##########
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

########## Torch packages ##########

from src.datasets.mvtec import load_mvtec_dataset
from src.losses.focal import FocalFrequencyLoss
from src.losses.losses import CustomLoss, SSIM_Loss
from src.model import UNet
from src.trains.train_unet import train_approach

def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.01,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )

    parser.add_argument(
        "-i", "--item", dest="item", type=str, default=None, help="Item to train"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    losses = {
        "ssim" : {
            "loss" : SSIM_Loss(data_range=255.0, channel=3, size_average=False, nonnegative_ssim=True),
            "weight" : 0.9

        },
        "focal" : {
            "loss" : FocalFrequencyLoss(),
            "weight" : 0.9

        },
        "mse" : {
            "loss" : MSELoss(),
            "weight" : 0.9

        },

    }

    loss_choice = 'focal'
    criterion = CustomLoss(
        losses=[losses[loss_choice]['loss']], weights=[losses[loss_choice]['weight']]
    )


    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    datasets, abnormal_items, gt_defect = load_mvtec_dataset()
    batch_size = 1
    val_percent = 0.1

    if args.item in datasets:
        item = args.item
        ################## Train ##################
        train = datasets[item]["train"]
        val = datasets[item]["test"]
        abnormal_item = abnormal_items[item]
    else:
        raise FileNotFoundError(f"{args.item} not Found in ({datasets.keys()})")

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    abn_loader: Dict[str, DataLoader] = {
        k: DataLoader(
            abnormal_item[k],
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        for k in abnormal_item
    }
    log_dir = os.path.join("runs", item, loss_choice)
    writer = SummaryWriter(log_dir=log_dir)

    train_approach(
        net=net,
        train_loader=train_loader,
        validation_loader=val_loader,
        anbnormal_loader=abn_loader,
        intput_key="noised",
        epochs=args.epochs,
        batch_size=args.batchsize,
        writer=writer,
        lr=args.lr,
        device=device,
        normalized=False,
        log_dir=log_dir,
        criterion=criterion, 
        item=item
    )
    # python src/trains/train_unet.py --item bottle -e 1
