import torch
import random

import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import image_models
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss

from utils import AverageMeter, configure_optimizers
from l0_image_folder import L0ImageFolder
from model_utils import get_model


def init_training(cfg, rank):
    """Initializes training

    Args:
        cfg (dict): Config of the run

    Returns:
        net,optimizer,aux_optimizer,criterion,train_dataloader,test_dataloader,lr_scheduler,last_epoch
    """

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    # Training Setup
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(cfg.patch_size), transforms.ToTensor()]
    )

    validation_transforms = transforms.Compose(
        [transforms.CenterCrop(cfg.patch_size), transforms.ToTensor()]
    )

    if cfg.use_l0_data:
        train_dataset = L0ImageFolder(
            root=cfg.dataset,
            seed=cfg.seed,
            test_over_total_percentage=cfg.l0_test_over_tot,
            valid_over_train_percentage=cfg.l0_validation_over_train,
            l0_format=cfg.l0_format,
            target_resolution_merged_m=cfg.l0_target_resolution_merged_m,
            preloaded=True,
            split="train",
            transform=train_transforms,
            geographical_split_tolerance=cfg.l0_train_test_tolerance,
        )
        validation_dataset = L0ImageFolder(
            root=cfg.dataset,
            seed=cfg.seed,
            test_over_total_percentage=cfg.l0_test_over_tot,
            valid_over_train_percentage=cfg.l0_validation_over_train,
            l0_format=cfg.l0_format,
            target_resolution_merged_m=cfg.l0_target_resolution_merged_m,
            preloaded=True,
            split="validation",
            transform=validation_transforms,
            geographical_split_tolerance=cfg.l0_train_test_tolerance,
        )
    else:
        train_dataset = ImageFolder(
            cfg.dataset, split="train", transform=train_transforms
        )
        validation_dataset = ImageFolder(
            cfg.dataset, split="test", transform=validation_transforms
        )

    device = "cuda:" + str(rank) if cfg.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda:" + str(rank)),
        pin_memory_device=device,
    )

    train_dataloader_iter = iter(train_dataloader)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda:" + str(rank)),
        pin_memory_device=device,
    )
    if cfg.use_l0_data:
        if cfg.l0_format == "raw":
            net = get_model(
                model=cfg.model,
                pretrained=cfg.pretrained,
                in_channels=1,
                quality=cfg.model_quality,
            )
        else:
            net = get_model(
                model=cfg.model,
                pretrained=cfg.pretrained,
                in_channels=13,
                quality=cfg.model_quality,
            )
    else:
        net = image_models[cfg.model](
            quality=cfg.model_quality, pretrained=cfg.pretrained
        )

    if cfg.use_l0_data:
        if cfg.l0_format == "raw":
            net = get_model(
                model=cfg.model,
                pretrained=cfg.pretrained,
                in_channels=1,
                quality=cfg.model_quality,
            )
        else:
            net = get_model(
                model=cfg.model,
                pretrained=cfg.pretrained,
                in_channels=13,
                quality=cfg.model_quality,
            )
    else:
        net = image_models[cfg.model](
            quality=cfg.model_quality, pretrained=cfg.pretrained
        )

    net = net.to(device)

    # if cfg.cuda and torch.cuda.device_count() > 1:
    #     print("Using multiple device CustomDataParallel")
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, cfg)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=cfg.lmbda)

    last_epoch = 0
    if cfg.checkpoint:  # load from previous checkpoint
        print("Loading", cfg.checkpoint)
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
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
        validation_dataloader,
        lr_scheduler,
        last_epoch,
        train_dataloader_iter,
    )


def train_one_batch(
    rank,
    model,
    criterion,
    train_dataloader,
    train_dataloader_iter: DataLoader,
    optimizer,
    aux_optimizer,
    batch_idx,
    clip_max_norm,
):
    """Trains the model on one batch

    Args:
        rank (int): Rank index
        model (torch.model): model to train
        criterion (): Loss criteration, see compressai docs
        train_dataloader (torch.dataloader): loader for training data
        train_dataloader_iter (iterator): current iterator on the loader
        optimizer (torch.optimizer): optimizer for gradients
        aux_optimizer (torch.optimizer): auxiliary loss optimizer
        batch_idx (int): index of current batch
        clip_max_norm (): gradient clipping thingy

    Returns:
        iterator: the updated training data loader
    """
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

    if batch_idx % 100 == 0:
        print(
            f"Rank {rank} - "
            f"Training batch {batch_idx}: ["
            f'Loss: {out_criterion["loss"].item():.3f} |'
            f'MSE loss: {out_criterion["mse_loss"].item():.3f} |'
            f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            f"Aux loss: {aux_loss.item():.2f}"
        )

    return train_dataloader_iter


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    """Train the mode for one epoch.

    Args:
        model (torch.model): model to train
        criterion (): Loss criteration, see compressai docs
        train_dataloader (torch.dataloader): loader for training data
        optimizer (torch.optimizer): optimizer for gradients
        aux_optimizer (torch.optimizer): auxiliary loss optimizer
        epoch (int): index of current epoch
        clip_max_norm (): gradient clipping thingy
    """
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


def test_epoch(rank, epoch, test_dataloader, model, criterion):
    """Test the model

    Args:
        rank (int): index of the rank
        epoch (int): index of current epoch
        test_dataloader (torch.dataloader): loader for test data
        model (torch.model): model to test
        criterion (): Loss criteration, see compressai docs

    Returns:
        float: avg loss
    """
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
        f"Rank {rank} - "
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def eval_test_set(
    rank,
    optimizer,
    batch_idx,
    net,
    criterion,
    test_losses,
    test_dataloader,
    local_time_at_test,
    paseos_instance,
    lr_scheduler,
    best_loss,
):
    """Evaluate the set

    Args:
        rank (int): Rank index
        optimizer (torch.optimizer): optimizer for gradients
        batch_idx (int): index of current batch
        net (torch.model): model to train
        criterion (): Loss criteration, see compressai docs
        test_losses (_type_): _description_
        teset_dataloader (torch.dataloader): loader for testing data
        local_time_at_test (float): local paseos time at testing
        paseos_instance (paseos): paseos instance of the rank
        lr_scheduler (torch.scheduler): lr scheduler
        best_loss (float): best achieved loss

    Returns:
        tuple: loss, whether best, best loss achieved
    """
    # print(f"Rank {rank} - Evaluating test set")
    # print(f"Rank {rank} - Previous learning rate: {optimizer.param_groups[0]['lr']}")
    # start = time.time()
    loss = test_epoch(rank, batch_idx, test_dataloader, net, criterion)
    test_losses.append(loss.item())
    local_time_at_test.append(paseos_instance._state.time)
    # print(f"Rank {rank} - Test evaluation took {time.time() - start}s")

    lr_scheduler.step(loss)
    # print(f"Rank {rank} - New learning rate: {optimizer.param_groups[0]['lr']}")

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)
    return loss, is_best, best_loss
