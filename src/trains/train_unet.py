import logging
import os
from typing import Dict

########## Torch packages ##########
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

########## Torch packages ##########
from tqdm import tqdm

from src.eval import eval_net
from src.losses.focal import FocalFrequencyLoss
from src.model import UNet


criterion2 = FocalFrequencyLoss()

def train_approach(
    net: UNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    anbnormal_loader: Dict[str, DataLoader],
    writer: SummaryWriter,
    device: torch.device,
    intput_key: str = "noise",
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 0.001,
    normalized: bool = False,
    log_dir: str = "./",
    criterion: nn.Module = criterion2,
    item : str = None
):

    total_params = sum(param.numel() for param in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_train = len(train_loader)
    n_val = len(validation_loader)
    latest_val_score = float("Inf")
    models_dir = os.path.join(log_dir, "models")

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Nb Parameters:   {total_params}
        N. Train Params: {trainable_params}
        
    """
    )

    os.makedirs(models_dir, exist_ok=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    global_step = 0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch[intput_key]
                true_masks = batch["output"]
                if imgs.shape[1] != net.n_channels:
                    logging.info(imgs.shape[1])
                    continue

                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
        global_step += 1
        val_score = eval_net(
            net,
            validation_loader,
            device,
            criterion=criterion,
            epoch=epoch,
            dir_=os.path.join(log_dir, "eval/good", str(epoch)),
            normalize=normalized,
        )

        scheduler.step(val_score)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logging.info("Validation SSIM loss: {}".format(val_score))
        writer.add_scalar("Loss/test", val_score, global_step)
        writer.add_images("images", imgs, global_step)
        for status in anbnormal_loader:

            abn_score = eval_net(
                net,
                anbnormal_loader[status],
                device,
                criterion=criterion,
                epoch=epoch,
                dir_=os.path.join(log_dir, f"eval/{status}", str(epoch)),
                normalize=normalized,
            )
            logging.info(f"Loss on abnormal data {item}/{status} : {abn_score}")

        if latest_val_score > val_score:
            latest_val_score = val_score
            path_checkpoint = os.path.join(models_dir, f"model_epoch_{item}.pth")
            torch.save(net.state_dict(), path_checkpoint)
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()
