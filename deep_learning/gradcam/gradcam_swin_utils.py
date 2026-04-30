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
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image

from PIL import Image
import matplotlib.pyplot as plt

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from gradcam_shared_utils import (
    _unwrap,
    _frame_id,
    _pool_class_frame,
    denorm,
    list_frames,
    plot_gradcam,
)

def load_swinv2(
    checkpoint_path,
    device,
    num_classes,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model = timm.create_model(
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in checkpoint["model_state_dict"].items()
    }
    
    model.load_state_dict(state_dict, strict=True)
        
    return model.to(device).eval()

def denorm_swin(input_tensor, mean, std):
    input_tensor = input_tensor.detach().cpu()
    mean = torch.tensor(mean, dtype=input_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(std, dtype=input_tensor.dtype).view(3, 1, 1)
    return (input_tensor * std + mean).clamp(0, 1)

def transform_swin():
    return transforms.Compose([
        transforms.Resize(272, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ])

def _swin_cam(model, input_tensor, predicted_class):
    model.eval()
    model.zero_grad(set_to_none=True)
    
    target_layer = model.layers[-1]
    
    activations, gradients = [], []
    
    def save_activation_and_gradients(module, inputs, output):
        output = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

    hook_handle = target_layer.register_forward_hook(save_activation_and_gradients)
    
    try:
        input_tensor = input_tensor.requires_grad_(True)

        logits = _unwrap(model(input_tensor))
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
            
        target_score = logits[0, predicted_class]
        target_score.backward()

        activation = activations[-1]
        gradient = gradients[-1]

        channel_weight = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (activation * channel_weight).sum(dim=3)
        cam = F.relu(cam).unsqueeze(1)
        
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        cam = F.interpolate(
            cam,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        
        return logits, cam.detach().cpu()

    finally:
        hook_handle.remove()
    
def gradcam_swin(
    model,
    address_folders,
    class_names,
    save_path=None,
):
    model.eval()
    device = next(model.parameters()).device
    
    frame_paths = list_frames(address_folders)[:64]
    
    configuration = resolve_data_config({}, model=model)
    #transform_swin = create_transform(**configuration)
    mean, std = configuration["mean"], configuration["std"] 

    predicted_class, predicted_confidence = _pool_class_frame(
        model,
        frame_paths,
        transform_swin,
    )
    
    predicted_class_name = (
        class_names[predicted_class]
        if class_names is not None else str(predicted_class)
    )
    
    print(f"Predicted class: {predicted_class_name}\n"
          f"Predicted confidence: {predicted_confidence:.4f}")

    overlay_images = []
    image_titles = []

    for frame_path in frame_paths:
        input_tensor = (
            transform_swin()(Image.open(frame_path).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )

        logits, cam = _swin_cam(model, input_tensor, predicted_class)
        
        # calculate probability per frame
        frame_probability = float(
            F.softmax(logits, dim=1)[0, predicted_class].detach().cpu().item()
        )

        # apply heatmap on original frame
        overlay_image = overlay_mask(
            to_pil_image(denorm_swin(input_tensor[0],mean=mean, std=std)),
            to_pil_image(cam.detach().cpu(), mode="F"),
            alpha=0.50,
        )
        
        # write predicted class and probability per frame
        overlay_images.append(overlay_image)
        image_titles.append(
            f"{frame_path.stem}\n"
            f"{predicted_class_name}={frame_probability:.3f}"
        )
    
    plot_gradcam(
        overlay_images,
        image_titles,
        save_path)
        
    return None
