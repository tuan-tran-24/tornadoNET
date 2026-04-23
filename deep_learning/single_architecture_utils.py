import time

import torch
from tqdm.auto import tqdm

from shared_utils import (
freeze_batchnorm_layers,
_forward_all_frames,
)

# =========================================================
# Training
# =========================================================

def train_one_epoch_single(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    colour="green",
    freeze_bn=False,
    ep=None,
    epochs=None,
):
    train_t0 = time.perf_counter()

    model.train()
    if freeze_bn:
        freeze_batchnorm_layers(model)

    running_loss = 0.0
    running_correct = 0.0
    running_total = 0

    desc = f"epoch {ep}/{epochs}" if ep is not None and epochs is not None else "train"
    bar = tqdm(train_loader, desc=desc, unit="address", colour=colour)

    for frames, label in bar:
        frames = frames[0]               # [K,3,H,W]
        label = label.to(device).long()  # [1]

        logits = _forward_all_frames(model, frames, device)  # [1,C]
        loss = criterion(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_correct += float((logits.argmax(dim=1) == label).float().item())
        running_total += 1

        bar.set_postfix({
            "training loss": f"{running_loss / max(1, running_total):.4f}",
            "training accuracy": f"{running_correct / max(1, running_total):.4f}",
            "steps": running_total,
        })

    train_time = time.perf_counter() - train_t0
    train_loss = running_loss / max(1, running_total)
    train_accuracy = running_correct / max(1, running_total)

    return train_loss, train_accuracy, train_time

# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def evaluate_single(
    model,
    device,
    validation_loader,
    criterion,
):
    model.eval()

    validation_loss_sum = 0.0
    validation_correct = 0.0
    validation_total = 0

    vbar = tqdm(validation_loader, desc="validation", unit="address", leave=False, colour="blue")

    for frames, label in vbar:
        frames = frames[0]
        label = label.to(device).long()

        logits = _forward_all_frames(model, frames, device)
        loss = criterion(logits, label)

        validation_loss_sum += float(loss.item())
        validation_correct += float((logits.argmax(dim=1) == label).float().item())
        validation_total += 1

        vbar.set_postfix({
            "Validation Loss": f"{validation_loss_sum / max(1, validation_total):.4f}",
            "Validation Accuracy": f"{validation_correct / max(1, validation_total):.4f}",
            "Steps": validation_total,
        })

    validation_loss = validation_loss_sum / max(1, validation_total)
    validation_accuracy = validation_correct / max(1, validation_total)

    return validation_loss, validation_accuracy