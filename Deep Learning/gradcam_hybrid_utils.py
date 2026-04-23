from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, List, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image

from PIL import Image
import matplotlib.pyplot as plt

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from cnn_stm_utils import cnn_stm
from cnn_lstm_utils import cnn_lstm

from gradcam_shared_utils import transform_conv
from gradcam_shared_utils import (
    _unwrap,
    _frame_id,
    _load_sequence,
    _norm01,
    _normq,
    denorm,
    list_frames,
    plot_gradcam,
)

def load_cnn_lstm(
    checkpoint_path,
    device,
    num_classes,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in checkpoint["model_state_dict"].items()
    }
    
    model = cnn_lstm(num_classes)                
    model.load_state_dict(state_dict, strict=True)
    
    return model.to(device).eval()

def load_cnn_stm(
    checkpoint_path,
    device,
    num_classes,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in checkpoint["model_state_dict"].items()
    }
    
    num_classes = int(state_dict["head_sequence.1.weight"].shape[0])
    d_model = int(state_dict["frame_projection.weight"].shape[0])
    max_frames = int(state_dict["positional_encoding"].shape[1] - 1)

    m = cnn_stm(
        num_classes=num_classes,
        d_model=d_model,
        n_head=8,
        stm_layers=2,
        dropout=0.2,
        max_frames=max_frames,
    )

    model = cnn_stm(num_classes)                
    model.load_state_dict(state_dict, strict=True)
    
    return model.to(device).eval()
    
def gradcam_hybrid(
    model,
    address_folders,
    class_names=None,
    save_path=None,
):
    """
    Make predictions on address folders, then compute GradCAM
    and apply overlays on selected frames
    """
    device = next(model.parameters()).device
    frame_paths = list_frames(address_folders)[:64]
    
    input_tensor = _load_sequence(frame_paths, transform_conv, device)
    input_tensor.requires_grad_(True)

    target_layer = model.backbone.layer4
    forward_fn = model.forward_sequence

    model.eval()
    model.zero_grad(set_to_none=True)

    activations, gradients = [], []

    def save_activation_and_gradients(module, inputs, output):
        output = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

    hook_handle = target_layer.register_forward_hook(save_activation_and_gradients)

    try:
        with torch.enable_grad():
            logits = _unwrap(forward_fn(input_tensor))
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            predicted_class = int(logits.argmax(dim=1).item())
            predicted_confidence = float(F.softmax(logits.detach(), dim=1)[0, predicted_class].item())
            predicted_class_name = (
                class_names[predicted_class]
                if class_names is not None else str(predicted_class)
            )
            
            if logits.size(1) > 1:
                top2_classes = logits.detach().topk(2, dim=1).indices[0]
                negative_class = (
                    int(top2_classes[1]) 
                    if int(top2_classes[0]) == predicted_class 
                    else int(top2_classes[0])
                )
                target_score = (logits[:, predicted_class] - 0.5 * logits[:, negative_class]).sum()
            else:
                target_score = logits[:, predicted_class].sum()

            model.zero_grad(set_to_none=True)
            target_score.backward()

        activation = activations[-1]
        gradient = gradients[-1]
        
        if activation.ndim == 5 and activation.size(0) == 1:
            activation = activation.squeeze(0)
        if gradient.ndim == 5 and gradient.size(0) == 1:
            gradient = gradient.squeeze(0)

        num_frames = len(frame_paths)
        num_cam_frames = activation.size(0)

        # match the number of class-activation maps with the number of input frames
        if num_cam_frames != num_frames:
            index = torch.linspace(
                0, num_cam_frames - 1,
                steps=num_frames,
                device=activation.device
            ).round().long()
            activation = activation[index]
            gradient = gradient[index]

        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * activation).sum(dim=1)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False).squeeze(1)
        cam = _normq(cam, q0=0.05, q1=0.95)

        # sequential weights
        sequential_weights = gradient.abs().mean(dim=(1, 2, 3))
        sequential_weights = sequential_weights / (sequential_weights.max() + 1e-8)
        sequential_weights = 0.70 + 0.30 * sequential_weights
        cam = _norm01(cam * sequential_weights[:, None, None])
        
        print(f"Predicted class: {predicted_class_name}\n"
              f"Predicted confidence: {predicted_confidence:.4f}")
        
        overlay_images = []
        image_titles = []
        for frame_index, frame_path in enumerate(frame_paths):    
            # apply heatmap on original frame
            overlay_image = overlay_mask(
                to_pil_image(denorm(input_tensor[0, frame_index])),
                to_pil_image(cam[frame_index].detach().cpu(), mode="F"),
                alpha=0.50,
            )
            
            # write predicted class and probability per frame            
            overlay_images.append(overlay_image)
            image_titles.append(
                f"{frame_path.stem}\n"
            )

    finally:
        hook_handle.remove()
        
    plot_gradcam(
        overlay_images,
        image_titles,
        save_path)

    return None