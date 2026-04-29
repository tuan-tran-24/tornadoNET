import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as confusion
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm.notebook import tqdm

@torch.no_grad()
def confusion_matrix(
    model,
    loader,
    class_names,
    device=None,
    title="Confusion Matrix",
    cmap="Blues",
    save_path: str | None = None,   
    dpi: int = 300,                 
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    y_true_list = []
    y_pred_list = []

    for frames, label in tqdm(loader):
        frames = frames[0].to(device, non_blocking=True)  # [K,3,H,W]
        y_true = int(label.item())

        logits = model(frames)            # [K,C]
        bag_logits = logits.mean(dim=0)   # [C]

        y_pred = int(bag_logits.argmax().item())

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    print(classification_report(
        y_true_list,
        y_pred_list,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0
    ))

    cm_counts = confusion(
        y_true_list, y_pred_list,
        labels=list(range(len(class_names)))
    )

    cm_norm = confusion(
        y_true_list, y_pred_list,
        labels=list(range(len(class_names))),
        normalize="true"
    )

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap=cmap, values_format=".2f", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # save confusion matrix
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")

    plt.show()

    return cm_counts, cm_norm
