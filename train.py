import torch
import random
import torch

import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import image_models
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss

from utils import AverageMeter, CustomDataParallel, configure_optimizers


def init_training(args):
    """Initializes training

    Args:
        args (dict): Passed arguments / parameters

    Returns:
        net,optimizer,aux_optimizer,criterion,train_dataloader,test_dataloader,lr_scheduler,last_epoch
    """

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Training Setup
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    train_dataloader_iter = iter(train_dataloader)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=1, pretrained=args.pretrained)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return (
        net,
        optimizer,
        aux_optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
        last_epoch,
        train_dataloader_iter,
    )


def train_one_batch(
    model,
    criterion,
    train_dataloader,
    train_dataloader_iter: DataLoader,
    optimizer,
    aux_optimizer,
    batch_idx,
    clip_max_norm,
):
    model.train()
    device = next(model.parameters()).device

    try:
        d = next(train_dataloader_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        train_dataloader_iter = iter(train_dataloader)
        d = next(train_dataloader_iter)

    d = d.to(device)
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    out_net = model(d)

    out_criterion = criterion(out_net, d)
    out_criterion["loss"].backward()
    if clip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()

    aux_loss = model.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()

    if batch_idx % 10 == 0:
        print(
            f"Training batch {batch_idx}: ["
            f'\tLoss: {out_criterion["loss"].item():.3f} |'
            f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            f"\tAux loss: {aux_loss.item():.2f}"
        )

    return train_dataloader_iter


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg
