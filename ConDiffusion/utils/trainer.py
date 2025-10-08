import numpy as np
import torch

from collections import defaultdict
from torch import Tensor
from tqdm import trange
from typing import *
import torch, torch.nn.functional as F
from functools import lru_cache
from math import cos, pi

from .sde import VPSDE

def get_category_dict(result_dict, categories, losses):
    # Iterate over each category and its corresponding value
    for category, loss in zip(categories, losses):
        # Check if the category is already a key in the dictionary
        if category.item() not in result_dict:
            # If not, add it with an empty list as the value
            result_dict[category.item()] = []
        # Append the value to the list associated with the category key
        result_dict[category.item()].append(loss.item())
    return dict(result_dict)


def loop(
        sde: VPSDE,
        trainset: Tensor,
        validset: Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int = 256,
        epoch_size: int = 4096,
        batch_size: int = 64,
        scheduler=None,
        device: str = "cpu",
        log_loss_per_level: bool = False,
        ema=None,
        data_spatial: int = 1,
        **absorb,
) -> Iterator:
    print("Model")
    print(sde)

    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train = []
        losses_valid = []
        losses_train_dict = defaultdict(list)
        losses_valid_dict = defaultdict(list)

        sde.train()
        idx = torch.randperm(trainset['data'].size(0), device='cpu')[:epoch_size]  # e.g., epoch_size=4096
        data_epoch = trainset['data'][idx]  # stay on CPU
        train_data = data_epoch.to(device).split(batch_size)


        for xb in train_data:
            optimizer.zero_grad()

            x_condition = xb[:, :-1, :].contiguous()
            x_target = xb[:, -1:, :].contiguous()
            losses, time_cat = sde.loss(x_target, x_condition, log_loss_per_level=log_loss_per_level)
            losses_train_dict = get_category_dict(losses_train_dict, time_cat, losses)
            l = losses.mean()

            l.backward()
            optimizer.step()

            if ema is not None:
                ema.update(sde.parameters())

            losses_train.append(l.detach())

        # Valid
        sde.eval()
        valid_data = validset['data'].to(device).split(batch_size)

        with torch.no_grad():
            for xb in valid_data:
                x_condition = xb[:, :-1, :].contiguous()
                x_target = xb[:, -1:, :].contiguous()
                losses, time_cat = sde.loss(x_target, x_condition, log_loss_per_level=log_loss_per_level)
                losses_valid_dict = get_category_dict(losses_valid_dict, time_cat, losses)
                losses_valid.append(losses.mean())

        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        lr = optimizer.param_groups[0]["lr"]

        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=lr)

        # step scheduler if provided
        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()

        losses_train_mean = []
        losses_valid_mean = []
        for key in range(11):
            if key in losses_train_dict.keys():
                losses_train_mean.append(np.mean(np.array(losses_train_dict[key])))
            else:
                losses_train_mean.append(0)
            if key in losses_valid_dict.keys():
                losses_valid_mean.append(np.mean(np.array(losses_valid_dict[key])))
            else:
                losses_valid_mean.append(0)
        yield loss_train, loss_valid, lr, losses_train_mean, losses_valid_mean, epoch
