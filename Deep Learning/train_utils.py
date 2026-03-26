import copy
import os
import random
import time
import re
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import clear_output
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from shared_utils import (
    _apply_aug_base,
    _load_rgb,
)

from single_architecture_utils import (
    train_one_epoch_single,
    evaluate_single,
)

from hybrid_architecture_utils import (
    train_one_epoch_hybrid,
    evaluate_hybrid,
)

# =========================================================
# Regex
# =========================================================

FRAME_RE = re.compile(r".*?_frame_(\d+)\.[A-Za-z0-9]+$", re.IGNORECASE)

# =========================================================
# Helpers functions
# =========================================================

def set_dropout(model, dropout_p: float):
    if hasattr(model, "fc"):   # ResNet50
        if not (
            isinstance(model.fc, nn.Sequential)
            and len(model.fc) > 0
            and isinstance(model.fc[0], nn.Dropout)
        ):
            model.fc = nn.Sequential(nn.Dropout(dropout_p), model.fc)
        else:
            model.fc[0].p = dropout_p

    elif hasattr(model, "head"):   # SwinV2-B
        if not (
            isinstance(model.head, nn.Sequential)
            and len(model.head) > 0
            and isinstance(model.head[0], nn.Dropout)
        ):
            model.head = nn.Sequential(nn.Dropout(dropout_p), model.head)
        else:
            model.head[0].p = dropout_p

def _drop_surrounding_in_meta(meta: dict) -> None:
    """Remove any frame paths that are under a 'Surrounding' subfolders"""
    surrounding = os.sep + "Surrounding" + os.sep
    for a in meta.get("train_addrs", []):
        a["frames"] = [p for p in a["frames"] if surrounding not in p]

def _build_address_list(root_dir: str, class_names: Optional[List[str]] = None):
    """ scan the training folder and turn it into meta['train_addrs'] list that DataLoader needs """
    
    root = Path(root_dir)
    
    if class_names is None:
        class_names = [d.name for d in root.iterdir() if d.is_dir()]
        class_names.sort()
        
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    addrs = []
    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
            
        for addr_dir in sorted([p for p in cls_dir.iterdir() if p.is_dir()]):
            # top-level frames only
            frames = []
            for f in addr_dir.iterdir():
                if f.is_file() and f.suffix.lower() in exts and FRAME_RE.match(f.name):
                    frames.append(str(f))
                    
            def frame_key(p: str):
                m = FRAME_RE.match(os.path.basename(p))
                return int(m.group(1)) if m else 10**18         
            
            frames.sort(key=frame_key)
            
            if frames:
                addrs.append({
                    "label": class_to_idx[cls],
                    "addr_dir": str(addr_dir),
                    "frames": frames,
                })
                
    return addrs, class_names

def _live_plot(history: dict):
    """Live-plot loss and accuracy per epoch"""
    xs = range(1, len(history["training loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 3.8))
    plt.plot(xs, history["training loss"], marker="o", label="Training Loss")

    if history.get("validation loss") and any(v is not None for v in history["validation loss"]):
        plt.plot(xs, history["validation loss"], marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.title("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 3.8))
    plt.plot(xs, history["training accuracy"], marker="s", label="Training Accuracy")

    if history.get("validation accuracy") and any(v is not None for v in history["validation accuracy"]):
        plt.plot(xs, history["validation accuracy"], marker="s", label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.title("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# Dataset
# =========================================================

class PrepareAddressFolder(Dataset):
    def __init__(
        self,
        addrs: List[Dict], 
        pairs: List[Tuple[int, bool]], # eg. [(1, True), (2, False),...etc.]
        aug_transform, 
        base_transform, 
        max_frames: Optional[int] = None,
    ):
        """ 
        Parameters:
        addrs: List of dictionaries; eg. {"label": 3, "frames":[".../frame_0001.jpg" , ".../frame_0002.jpg", ...]}])
        pairs: List of (address index, augmentation flag), e.g. [(1, True), (2, False), ...]
        aug_tranform: Albumentation transforms
        base_transform: resize, normalize, tensorize
        """
    
        self.addrs = addrs
        self.pairs = pairs
        self.aug_t = aug_transform
        self.base_t = base_transform
        self.max_frames = max_frames

    def __len__(self):
        """ return the number of pairs """
        return len(self.pairs)

    def __getitem__(self, i):
        """ Return (frames tensor, label) """
        addr_idx, use_aug = self.pairs[i]
        entry  = self.addrs[addr_idx]
        label  = int(entry["label"])

        imgs = []

        frames_list = entry["frames"]

        if self.max_frames is not None and len(frames_list) > self.max_frames:
            frames_list = frames_list[:self.max_frames]   # keeps order

        do_aug = bool(use_aug)
        
        for fp in frames_list:
            img_np = _load_rgb(fp)
            out = _apply_aug_base(img_np, do_aug, self.aug_t, self.base_t)
            imgs.append(out)
       
        # stack all frames into one tensor and return in the format of [K, 3, H, W]
        # where, K = number of frames in the address folder
        frames = torch.stack(imgs, dim=0)  
        return frames, torch.tensor(label).long()

# =========================================================
# DataLoader builder
# =========================================================

def build_train_loader(
    augmentation_ds,        # holds transforms from aug_t and base_t          
    meta: dict,
    batch_size: int = 1,      # keep as 1 because each folder has a variable number of frames
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_frames: Optional[int] = None,
):
    if batch_size != 1:
        # Pytorch can't stack different K into a batch tensor. Hence, batch size must be kept as 1
        raise ValueError("batch_size must be 1 because each address has variable number of frames.")

    if "train_addrs" not in meta or "train_pairs" not in meta:
        raise KeyError("meta must contain 'train_addrs' and 'train_pairs'.")
    
    # remove all paths in the "Surrounding" subfolders
    _drop_surrounding_in_meta(meta)

    addrs = meta["train_addrs"] 
    pairs = meta["train_pairs"]

    # obtain augmentation transforms from 'augmentation_ds'
    aug_t  = augmentation_ds["aug_t"]
    base_t = augmentation_ds["base_t"]

    if aug_t is None or base_t is None:
        raise ValueError("Expected augmentation_ds to have aug/base transforms.")

    ds = PrepareAddressFolder(
        addrs=addrs,
        pairs=pairs,
        aug_transform=aug_t,
        base_transform=base_t,
        max_frames=max_frames,
    )

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

def sample_addresses_per_epoch(
    addrs,
    class_names,
    seed,
    epoch,
    class_sampling=None,
):
    idx_by_y = {}
    for addr_idx, a in enumerate(addrs):
        idx_by_y.setdefault(int(a["label"]), []).append(addr_idx)

    rng = random.Random(seed + epoch)
    keep_addr = set()

    for y, idxs in idx_by_y.items():
        class_name = class_names[y]
        rule = class_sampling.get(class_name, "all") if class_sampling is not None else "all"

        if rule == "all" or rule is None:
            keep_addr |= set(idxs)
        else:
            keep_addr |= set(rng.sample(idxs, min(int(rule), len(idxs))))

    return sorted(keep_addr)

def build_train_pairs_per_epoch(
    addrs,
    class_names,
    seed,
    epoch,
    class_sampling=None,
    augmentation_factors=None,
):
    keep_addr = sample_addresses_per_epoch(
        addrs=addrs,
        class_names=class_names,
        seed=seed,
        epoch=epoch,
        class_sampling=class_sampling,
    )

    rng = random.Random(seed + 10000 + epoch)
    train_pairs = []

    for addr_idx in keep_addr:
        y = int(addrs[addr_idx]["label"])
        class_name = class_names[y]

        if augmentation_factors is None:
            aug_factor = 1
        else:
            aug_factor = max(1, int(augmentation_factors.get(class_name, 1)))

        train_pairs.append((addr_idx, False))

        for _ in range(aug_factor - 1):
            train_pairs.append((addr_idx, True))

    rng.shuffle(train_pairs)
    return train_pairs, keep_addr

# =========================================================
# Training
# =========================================================

def train_model(
    model,
    epochs: int,
    optimizer,
    mode: str,   # "single" or "hybrid"
    lr: Union[float, List[float]] = 1e-4,
    lr_epochs: Optional[List[int]] = None,
    device: Optional[str] = None,
    colour: str = "green",
    validation_loader=None,
    csv_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: str = "model",
    freeze_bn: bool = False,
    plot_fn=None,
    class_sampling: Optional[dict] = None,
    augmentation_factors: Optional[dict] = None,

    # single-stream args
    augmentation_ds=None,
    meta: Optional[dict] = None,
    class_names=None,
    seed: int = 42,
    dropout_p: Optional[float] = None,
    apply_dropout: bool = False,

    # hybrid args
    sequence_loader: Optional[DataLoader] = None,
    still_loader: Optional[DataLoader] = None,
    w_sequence: float = 0.6,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if optimizer is None:
        raise ValueError("optimizer must not be None")

    if mode not in ["single", "hybrid"]:
        raise ValueError("mode must be 'single' or 'hybrid'")

    if apply_dropout and dropout_p is not None:
        set_dropout(model, dropout_p)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "training loss": [],
        "training accuracy": [],
        "validation loss": [],
        "validation loss ma": [],
        "validation accuracy": [],
        "epoch time": [],
        "training time": [],
        "validation time": [],
    }

    log_rows = []

    best_raw_validation = float("inf")
    best_raw_epoch = None
    best_raw_state = None

    best_ma_validation = float("inf")
    best_ma_epoch = None
    best_ma_state = None
    
    epoch_state_buffer = []
    
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        # LR schedule
        if isinstance(lr, (list, tuple)):
            if lr_epochs is None or len(lr_epochs) != len(lr):
                raise ValueError("If lr is list/tuple, provide lr_epochs with same length.")
            current_lr = float(lr[0])
            for e_start, lr_val in zip(lr_epochs, lr):
                if ep >= e_start:
                    current_lr = float(lr_val)
        else:
            current_lr = float(lr)

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # -------------------------
        # build epoch data + train
        # -------------------------
        if mode == "single":
            addrs = meta["train_addrs"]

            meta["train_pairs"], keep_addr = build_train_pairs_per_epoch(
                addrs=addrs,
                class_names=class_names,
                seed=seed,
                epoch=ep,
                class_sampling=class_sampling,
                augmentation_factors=augmentation_factors,
            )
            
            
            steps_this_epoch = len(meta["train_pairs"])

            train_loader = build_train_loader(
                augmentation_ds=augmentation_ds,
                meta=meta,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                max_frames=64,
            )

            train_loss, train_accuracy, train_time = train_one_epoch_single(
                model=model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                colour=colour,
                freeze_bn=freeze_bn,
                ep=ep,
                epochs=epochs,
            )

            row_extra = {
                "steps": steps_this_epoch,
            }

        else:
            sequence_ds = sequence_loader.dataset
            still_ds = still_loader.dataset
            addrs = sequence_ds.addrs

            keep_addr = sample_addresses_per_epoch(
                addrs=addrs,
                class_names=class_names,
                seed=seed,
                epoch=ep,
                class_sampling=class_sampling,
            )
            
            addr_repeat = {}
            for addr_idx in keep_addr:
                y = int(addrs[addr_idx]["label"])
                class_name = class_names[y]
        
                if augmentation_factors is None:
                    repeat = 1
                else:
                    repeat = max(1, int(augmentation_factors.get(class_name, 1)))
        
                addr_repeat[addr_idx] = repeat
        
            sequence_idx = []
            for i, s in enumerate(sequence_ds.samples):
                addr_idx = s[0]
                repeat = addr_repeat.get(addr_idx, 0)
                for _ in range(repeat):
                    sequence_idx.append(i)
        
            still_idx = []
            for i, s in enumerate(still_ds.samples):
                addr_idx = s[0]
                repeat = addr_repeat.get(addr_idx, 0)
                for _ in range(repeat):
                    still_idx.append(i)
            
            sequence_loader_ep = DataLoader(
                sequence_ds,
                batch_size=sequence_loader.batch_size,
                sampler=SubsetRandomSampler(sequence_idx),
                num_workers=sequence_loader.num_workers,
                pin_memory=sequence_loader.pin_memory,
                drop_last=False,
            )

            still_loader_ep = DataLoader(
                still_ds,
                batch_size=still_loader.batch_size,
                sampler=SubsetRandomSampler(still_idx),
                num_workers=still_loader.num_workers,
                pin_memory=still_loader.pin_memory,
                drop_last=False,
            )

            steps_this_epoch = max(len(sequence_loader_ep), len(still_loader_ep))

            train_loss, train_accuracy, train_time = train_one_epoch_hybrid(
                model=model,
                device=device,
                sequence_loader_ep=sequence_loader_ep,
                still_loader_ep=still_loader_ep,
                optimizer=optimizer,
                criterion=criterion,
                w_sequence=w_sequence,
                colour=colour,
                freeze_bn=freeze_bn,
                ep=ep,
                epochs=epochs,
            )

            row_extra = {
                "steps": steps_this_epoch,
                "keep_addr_count": len(keep_addr),
                "sequence_samples": len(sequence_idx),
                "still_samples": len(still_idx),
                "w_sequence": w_sequence,
            }

        history["training loss"].append(train_loss)
        history["training accuracy"].append(train_accuracy)

        # -------------------------
        # validation
        # -------------------------
        validation_time = 0.0
        if validation_loader is not None:
            validation_t0 = time.perf_counter()

            if mode == "single":
                validation_loss, validation_accuracy = evaluate_single(
                    model=model,
                    device=device,
                    validation_loader=validation_loader,
                    criterion=criterion,
                )
            else:
                validation_loss, validation_accuracy = evaluate_hybrid(
                    model=model,
                    device=device,
                    validation_loader=validation_loader,
                    criterion=criterion,
                    w_sequence=w_sequence,
                )

            validation_time = time.perf_counter() - validation_t0
        else:
            validation_loss = None
            validation_accuracy = None

        history["validation loss"].append(validation_loss)
        history["validation accuracy"].append(validation_accuracy)

        # -------------------------
        # best validation loss based on raw score
        # -------------------------
        if validation_loss is not None and validation_loss < best_raw_validation:
            best_raw_validation = float(validation_loss)
            best_raw_epoch = ep
            best_raw_state = copy.deepcopy(model.state_dict())
        
        # keep rolling states for centered MA selection
        epoch_state_buffer.append({
            "epoch": ep,
            "state_dict": copy.deepcopy(model.state_dict()),
        })
        
        if len(epoch_state_buffer) > 5:
            epoch_state_buffer.pop(0)
        
        # -------------------------
        # centered 5-point moving average
        # -------------------------
        validation_ma = None
        history["validation loss ma"].append(None)

        if len(history["validation loss"]) >= 5 and len(epoch_state_buffer) >= 5:
            win = history["validation loss"][-5:]
            if all(v is not None for v in win):
                validation_ma = float(sum(win) / 5.0)

                # centered MA belongs to the middle epoch in the 5-epoch window
                history["validation loss ma"][-3] = validation_ma

                if validation_ma < best_ma_validation:
                    center_item = epoch_state_buffer[2]   # middle of 5 buffered epochs
                    best_ma_validation = float(validation_ma)
                    best_ma_epoch = center_item["epoch"]
                    best_ma_state = copy.deepcopy(center_item["state_dict"])
        
        # selected best
        if best_ma_state is not None:
            best_state = best_ma_state
            best_epoch = best_ma_epoch
            best_validation_loss_selected = best_ma_validation
        else:
            best_state = best_raw_state
            best_epoch = best_raw_epoch
            best_validation_loss_selected = (
                best_raw_validation if best_raw_state is not None else None
            )        
            
        epoch_time = time.perf_counter() - epoch_t0
        history["epoch time"].append(epoch_time)
        history["training time"].append(train_time)
        history["validation time"].append(validation_time)

        row = {
            "epoch": ep,
            **row_extra,
            "training loss": train_loss,
            "training accuracy": train_accuracy,
            "validation loss": validation_loss,
            "validation loss ma": validation_ma,
            "validation accuracy": validation_accuracy,
            "epoch time": epoch_time,
            "training time": train_time,
            "validation time": validation_time,
            "learning rate": current_lr,
            "best epoch raw": best_raw_epoch,
            "best validation loss raw": best_raw_validation if best_raw_epoch is not None else None,
            "best epoch ma": best_ma_epoch,
            "best validation loss ma": best_ma_validation if best_ma_epoch is not None else None,
        }
        log_rows.append(row)

        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_epoch_{ep:03d}.pth")
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": train_loss,
                "training_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_loss_ma": validation_ma,
                "validation_accuracy": validation_accuracy,
                "learning_rate": current_lr,
                "best_epoch_raw": best_raw_epoch,
                "best_validation_loss_raw": best_raw_validation if best_raw_epoch is not None else None,
                "best_epoch_ma": best_ma_epoch,
                "best_validation_loss_ma": best_ma_validation if best_ma_epoch is not None else None,
            }, ckpt_path)

        if csv_path is not None:
            pd.DataFrame(log_rows).to_csv(csv_path, index=False)

        if plot_fn is not None:
            clear_output(wait=True)
            plot_fn(history)

    if best_state is None:
        best_state = best_raw_state
        best_ma_epoch = best_raw_epoch
        best_ma_validation = best_raw_validation

    return {
        "history": history,
        "log": pd.DataFrame(log_rows),
        "best_state_dict": best_state,
        "best_epoch": best_ma_epoch,
        "best_validation_loss_ma": best_ma_validation,
        "best_raw_state_dict": best_raw_state,
        "best_raw_epoch": best_raw_epoch,
        "best_raw_validation_loss": best_raw_validation,
    }