import os, random
from collections import Counter
from typing import Dict, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import matplotlib as mpl
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

def cnn_only_augmentation():
    minority_aug = A.Compose([
        A.Rotate(limit=12, p=0.35),         
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.10, hue=0.06,
            p=0.40
        ),
        A.GaussNoise(std_range=(0.01,0.05), p=0.15),
        A.MotionBlur(blur_limit=3, p=0.12),
        A.ToGray(p=0.05),
    ])

    base = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return minority_aug, base
    
def swin_augmentation():
    minority_aug = A.Compose([
        A.Rotate(limit=12, p=0.35),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.06, p=0.40),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),  
        A.MotionBlur(blur_limit=3, p=0.12),
        A.ToGray(p=0.05),
    ])

    base = A.Compose([
        A.SmallestMaxSize(272, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return minority_aug, base
def cnn_hybrid_augmentation():

    seq_aug = A.Compose([
        A.Rotate(limit=12, p=0.35),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.06, p=0.40),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        A.MotionBlur(blur_limit=3, p=0.12),
        A.ToGray(p=0.05),
        A.Resize(256, 256),
        # NO Normalize, NO ToTensorV2
    ])

    # slightly lighter for stills (less motion blur / less aggressive)
    still_aug = A.Compose([
        A.Rotate(limit=10, p=0.25),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.05, p=0.30),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.10),
        A.ToGray(p=0.03),
        A.Resize(256, 256),
        # NO Normalize, NO ToTensorV2
    ])

    base = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    return seq_aug, still_aug, base