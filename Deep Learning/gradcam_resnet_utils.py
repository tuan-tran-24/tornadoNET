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

from gradcam_shared_utils import transform_conv
from gradcam_shared_utils import (
    _unwrap,
    _frame_id,
    _pool_class_frame,
    denorm,
    list_frames,
    plot_gradcam,
)


def load_resnet50(
    checkpoint_path,
    device,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    num_classes = int(state_dict["fc.1.weight"].shape[0])

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, num_classes),
    )

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    return model, num_classes

def gradcam_resnet50(
    model,
    address_folders,
    class_names,
    save_path=None,
):
    """
    Make predictions on address folders, then compute GradCAM
    and apply overlays on selected frames
    """      
    model.eval()
    device = next(model.parameters()).device
    
    frame_paths = list_frames(address_folders)[:64]
    
    predicted_class, predicted_confidence = _pool_class_frame(
        model,
        frame_paths,
    )
    
    predicted_class_name = (
        class_names[predicted_class]
        if class_names is not None else str(predicted_class)
    )
    
    print(f"Predicted class: {predicted_class_name}\n"
          f"Predicted confidence: {predicted_confidence:.4f}")

    gradcam_extractor = GradCAM(model, target_layer=model.layer4)
    
    overlay_images = []
    image_titles = []
    
    for frame_path in frame_paths:
        input_tensor = (
            transform_conv()(Image.open(frame_path).convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )
        input_tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = _unwrap(model(input_tensor))
        
        cam = gradcam_extractor(
            class_idx=predicted_class, 
            scores=logits,
        )[0]
        
        if cam.ndim == 3:
            cam = cam[0]

        # calculate probability per frame
        frame_probability = float(
            F.softmax(logits, dim=1)[0, predicted_class].detach().cpu().item()
        )

        # apply heatmap on original frame
        overlay_image = overlay_mask(
            to_pil_image(denorm(input_tensor[0])),
            to_pil_image(cam.detach().cpu(), mode="F"),
            alpha=0.50,
        )

        # write predicted class and probability per frame
        overlay_images.append(overlay_image)
        image_titles.append(
            f"{frame_path.stem}\n"
            f"{predicted_class_name}={frame_probability:.3f}"
        )

    try:
        gradcam_extractor.remove_hooks()
    except Exception:
        pass   
    
    plot_gradcam(
        overlay_images,
        image_titles,
        save_path)
        
    return None