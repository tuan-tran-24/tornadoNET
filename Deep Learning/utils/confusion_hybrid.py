import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as confusion
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm

@torch.no_grad()
def confusion_matrix(
    model: torch.nn.Module,
    device: str,
    class_names: list[str],
    validation_loader,
    w_sequence: float = 0.6,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: str | None = None,   
    dpi: int = 300,                 
):
    model.eval()
    model.to(device)

    w_still = 1.0 - w_sequence

    y_true_list = []
    y_pred_list = []

    for batch in tqdm(validation_loader, desc="eval addrs", leave=False):
        y = int(batch["y"])
        y_t = torch.tensor([y], device=device).long()  

        have_sequence = False
        have_still = False
        logits_sequence = None
        logits_still = None

        sequence_runs = batch.get("seq_runs", [])
        if sequence_runs is not None and len(sequence_runs) > 0:
            logits_runs = []
            run_lens = []

            for run in sequence_runs:
                x = run.unsqueeze(0).to(device, non_blocking=True)
                logits_runs.append(model.forward_sequence(x))  # [1,C]
                run_lens.append(run.shape[0])

            w = torch.tensor(run_lens, device=device, dtype=torch.float32)
            w = w / w.sum().clamp_min(1.0)

            logits_runs = torch.stack(logits_runs, dim=0).squeeze(1)  # [R,C]
            logits_sequence = (logits_runs * w.unsqueeze(1)).sum(dim=0, keepdim=True)  # [1,C]
            have_sequence = True

        still = batch.get("still", None)
        if still is not None:
            xs = still.unsqueeze(0).to(device, non_blocking=True)  # [1,K,3,H,W]
            logits_still = model.forward_still(xs)                 # [1,C]
            have_still = True

        if have_sequence and have_still:
            logits = w_sequence * logits_sequence + w_still * logits_still
        elif have_sequence:
            logits = logits_sequence
        elif have_still:
            logits = logits_still
        else:
            continue

        probs = torch.softmax(logits.squeeze(0), dim=0)
        _, y_pred_t = probs.max(dim=0)
        y_pred = int(y_pred_t.item())
        
        y_true_list.append(y)
        y_pred_list.append(y_pred)

    y_true_arr = np.array(y_true_list, dtype=int)
    y_pred_arr = np.array(y_pred_list, dtype=int)

    print(classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0
    ))

    cm_counts = confusion(
        y_true_arr, y_pred_arr,
        labels=list(range(len(class_names)))
    )
    cm_norm = confusion(
        y_true_arr, y_pred_arr,
        labels=list(range(len(class_names))),
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap=cmap, values_format=".2f", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # save PNG
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved confusion matrix PNG to: {save_path}")

    plt.show()

    return cm_counts, cm_norm
