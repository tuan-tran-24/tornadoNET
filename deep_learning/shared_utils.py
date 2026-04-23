import os
import re
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch import nn

def _replay_from(comp: A.Compose) -> A.ReplayCompose:
    """ Turns an Albumentations Compose into Replay Compose, so exact augmentation is done on the sequence"""
    return A.ReplayCompose([copy.deepcopy(t) for t in comp.transforms])

def freeze_batchnorm_layers(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def _load_rgb(path: str) -> np.ndarray:
    """ Load an image file and return a NumPy array in RGB format (H, W, 3). """
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))

def _apply_aug_base(img_np, do_aug, aug_t, base_t):
    if do_aug:
        img_np = aug_t(image=img_np)["image"]   
    out = base_t(image=img_np)["image"]         
    if not torch.is_tensor(out):
        raise TypeError("base_t must output a torch.Tensor")
    return out

def _forward_all_frames(
    model: torch.nn.Module,
    frames: torch.Tensor,                 
    device: str
) -> torch.Tensor:
    """ This function takes all frames from an address folder and turns them into one set of logit"""
    # move frames to GPU/CPU
    frames = frames.to(device, non_blocking=True) # [K, 3, H, W]

    # run the model on K frames
    out = model(frames)                               # [K, C]

    # global averaging across all frames
    bag_logits = out.mean(dim=0, keepdim=True)        # [1, C]
    return bag_logits