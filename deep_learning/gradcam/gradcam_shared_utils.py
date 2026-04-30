from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, List, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# ============================================================
# Helper functions
# ============================================================

def _unwrap(y):
    return y[0] if isinstance(y, (tuple, list)) else y

def transform_conv():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def denorm(input_tensor):
    input_tensor = input_tensor.detach().cpu()
    mean = torch.tensor((0.5, 0.5, 0.5), dtype=input_tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5), dtype=input_tensor.dtype).view(3, 1, 1)
    return (input_tensor * std + mean).clamp(0, 1)

def _frame_id(path):
    stem = path.stem

    for pattern in [r"_frame_(\d+)$", r"frame[_\-\s]?(\d+)"]:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        numbers = re.findall(r"\d+", stem)
        return int(numbers[-1]) if numbers else -1

def list_frames(address_folders):
    img_externals = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    root = Path(address_folders)
    
    frame_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in img_externals
        and not any(parent.name.lower() == "surrounding" for parent in path.parents)
    ]

    frame_paths.sort(key=lambda path: (_frame_id(path), path.name.lower()))
    return frame_paths

def _load_sequence(
    frame_paths,
    transform,
    device,
):
    input_tensors = [transform()(Image.open(frame_path).convert("RGB")) for frame_path in frame_paths]
    input_tensor = torch.stack(input_tensors, dim=0).unsqueeze(0).to(device)  # [1,T,3,H,W]
    return input_tensor

@torch.no_grad()
def _pool_class_frame(
    model,
    frame_paths,
    transform=None,
):
    if transform is None:
        transform = transform_conv
        
    device = next(model.parameters()).device
    selected_frames = frame_paths[:64]
    
    sum_logits = None
    for frame_path in selected_frames:
        image_tensor = (
            transform()(Image.open(frame_path).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )
        
        logits = _unwrap(model(image_tensor))

        if sum_logits is None:
            sum_logits = logits
        else:
            sum_logits += logits

    mean_logit = sum_logits / len(selected_frames)
    probabilities = F.softmax(mean_logit, dim=1)[0]
    
    predicted_class = int(probabilities.argmax().item())
    predicted_confidence = float(probabilities[predicted_class].item())
    return predicted_class, predicted_confidence

# ============================================================
# Visualization
# ============================================================

def _norm01(input_tensor, q0=0.05, q1=0.95, eps=1e-8):
    flat_tensor = input_tensor.view(input_tensor.size(0), -1)
    
    minmin = flat_tensor.min(dim=1).values.view(-1, 1, 1)
    maxmax = flat_tensor.max(dim=1).values.view(-1, 1, 1)
    
    return (input_tensor - minmin) / (maxmax - minmin + eps)

def _normq(input_tensor, q0=0.05, q1=0.95, eps=1e-8):
    flat_tensor = input_tensor.view(input_tensor.size(0), -1)
    
    low = torch.quantile(flat_tensor, q0, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat_tensor, q1, dim=1).view(-1, 1, 1)
    
    normalized = (input_tensor - low) / (high - low + eps)
    return normalized.clamp(0, 1)

def plot_gradcam(
    overlay_images,
    titles=None,
    save_path=None,
):
    overlay_images = overlay_images[:4] # set max number of frames to show
    num_images = len(overlay_images)
    num_columns = min(6, num_images)
    num_rows = (num_images + num_columns - 1) // num_columns
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3))
    axes= axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax in axes:
        ax.axis("off")
        
    for i in range(num_images):
        axes[i].imshow(overlay_images[i])
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=8)

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
        )
        
    plt.show()
