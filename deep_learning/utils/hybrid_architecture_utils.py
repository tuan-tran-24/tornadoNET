import os
import random
import re
import time
from typing import Dict, List, Tuple

import albumentations as A
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from shared_utils import (
freeze_batchnorm_layers,
_load_rgb,
_replay_from,
)

# =========================================================
# Regex
# =========================================================

FRAME_RE = re.compile(r".*?_frame_(\d+)\.[A-Za-z0-9]+$", re.IGNORECASE)

# =========================================================
# Helpers functions
# =========================================================

def unwrap_batch(batch):
    return batch[0]

def build_sequence_and_still(
    addrs: List[Dict],
    run_min: int = 3,
    max_sequence_len: int | None = 20,
    max_still_frames: int | None = 64,
):
    """
    For each address:
      - Find consecutive runs (>= run_min) -> sequence samples
      - Remaining frames -> still bag
    Caps:
      - each sequence run capped to first max_seq_len frames
      - still bag capped to first max_still_frames frames
    """
    sequence_runs: List[Tuple[int, List[str]]] = []
    still_runs: Dict[int, List[str]] = {}

    for i, a in enumerate(addrs):
        pairs = []
        for p in a["frames"]:
            m = FRAME_RE.match(os.path.basename(p))
            if m:
                pairs.append((int(m.group(1)), p))

        if not pairs:
            continue

        pairs.sort(key=lambda x: x[0])

        # split into consecutive runs
        runs, cur = [], []
        for idx, p in pairs:
            if not cur or idx == cur[-1][0] + 1:
                cur.append((idx, p))
            else:
                runs.append(cur)
                cur = [(idx, p)]
        if cur:
            runs.append(cur)

        used = set()

        # sequences
        for run in runs:
            if len(run) >= run_min:
                paths = [p for _, p in run]
                if max_sequence_len is not None and len(paths) > max_sequence_len:
                    paths = paths[:max_sequence_len]   # keep order
                sequence_runs.append((i, paths))
                used.update(paths)

        # still leftovers
        leftovers = [p for _, p in pairs if p not in used]
        if max_still_frames is not None and len(leftovers) > max_still_frames:
            leftovers = leftovers[:max_still_frames]  # keep order
        if leftovers:
            still_runs[i] = leftovers

    return sequence_runs, still_runs

# =========================================================
# Set up sequence and still datasets
# =========================================================

class SequenceDataset(Dataset):
    def __init__(self, sequence_runs, addrs, aug_t, base_t, repeat_factors):
        self.addrs = addrs
        self.aug_t = aug_t
        self.base_t = base_t
        self.repeat_factors = repeat_factors
        self.samples = []

        for addr_idx, paths in sequence_runs:
            y = int(addrs[addr_idx]["label"])
            factor = int(repeat_factors.get(y, 1))

            self.samples.append((addr_idx, paths, False))
            for _ in range(max(0, factor - 1)):
                self.samples.append((addr_idx, paths, True))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        addr_idx, paths, use_aug = self.samples[i]
        y = int(self.addrs[addr_idx]["label"])

        if not use_aug:
            tensors = [self.base_t(image=_load_rgb(p))["image"] for p in paths]
        else:
            rc = _replay_from(self.aug_t)
            out0 = rc(image=_load_rgb(paths[0]))
            rep = out0["replay"]

            tensors = [self.base_t(image=out0["image"])["image"]]
            for p in paths[1:]:
                x = A.ReplayCompose.replay(rep, image=_load_rgb(p))["image"]
                tensors.append(self.base_t(image=x)["image"])

        sequence = torch.stack(tensors, dim=0)  # [T,3,H,W]
        return sequence, torch.tensor(y).long()

class StillDataset(Dataset):
    def __init__(self, leftovers_by_addr, addrs, aug_t, base_t, repeat_factors, seed=42):
        self.left = leftovers_by_addr
        self.addrs = addrs
        self.aug_t = aug_t
        self.base_t = base_t
        self.repeat_factors = repeat_factors
        rng = random.Random(seed)

        self.samples = []
        for addr_idx in self.left.keys():
            y = int(addrs[addr_idx]["label"])
            factor = int(repeat_factors.get(y, 1))

            self.samples.append((addr_idx, False))
            for _ in range(max(0, factor - 1)):
                self.samples.append((addr_idx, True))

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        addr_idx, use_aug = self.samples[i]
        y = int(self.addrs[addr_idx]["label"])

        imgs = []
        for p in self.left[addr_idx]:
            img = _load_rgb(p)
            if use_aug:
                img = self.aug_t(image=img)["image"]   # numpy
            imgs.append(self.base_t(image=img)["image"])  # ALWAYS base_t -> torch

        still = torch.stack(imgs, dim=0)  # [K,3,H,W]
        return still, torch.tensor(y).long()

class ValidationDataset(Dataset):
    def __init__(
        self,
        validation_addrs: list,
        validation_sequence_runs: list,
        validation_still_runs: dict,
        base_t,
        addr_indices: list | None = None,
    ):
        self.addrs = validation_addrs
        self.base_t = base_t

        # group runs by address
        self.sequence_by_addr = {}
        for addr_idx, paths in validation_sequence_runs:
            self.sequence_by_addr.setdefault(addr_idx, []).append(paths)

        self.still_by_addr = validation_still_runs

        if addr_indices is None:
            self.addr_indices = sorted(set(self.sequence_by_addr.keys()) | set(self.still_by_addr.keys()))
        else:
            self.addr_indices = list(addr_indices)

    def __len__(self):
        return len(self.addr_indices)

    def __getitem__(self, i):
        addr_idx = self.addr_indices[i]
        y = int(self.addrs[addr_idx]["label"])

        # sequences 
        sequence_runs = []
        for paths in self.sequence_by_addr.get(addr_idx, []):
            imgs = []
            for p in paths:
                img = _load_rgb(p)                         
                imgs.append(self.base_t(image=img)["image"]) # torch [3,H,W]
            if len(imgs) > 0:
                sequence_runs.append(torch.stack(imgs, dim=0))   # [T,3,H,W]

        # still bag
        still_paths = self.still_by_addr.get(addr_idx, [])
        still = None
        if len(still_paths) > 0:
            imgs = []
            for p in still_paths:
                img = _load_rgb(p)
                imgs.append(self.base_t(image=img)["image"])
            if len(imgs) > 0:
                still = torch.stack(imgs, dim=0)            # [K,3,H,W]

        return {"y": y, "sequence_runs": sequence_runs, "still": still}

def build_validation_loader(
    validation_addrs,
    validation_sequence_runs,
    validation_still_runs,
    validation_base_t,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    ds = ValidationDataset(
        validation_addrs=validation_addrs,
        validation_sequence_runs=validation_sequence_runs,
        validation_still_runs=validation_still_runs,
        base_t=validation_base_t,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=unwrap_batch,          
        persistent_workers=(num_workers > 0),
    )
    return loader

# =========================================================
# Training
# =========================================================

def train_one_epoch_hybrid(
    model,
    device,
    sequence_loader_ep,
    still_loader_ep,
    optimizer,
    criterion,
    w_sequence=0.6,
    colour="green",
    freeze_bn=False,
    ep=None,
    epochs=None,
):
    train_t0 = time.perf_counter()

    model.train()
    if freeze_bn:
        freeze_batchnorm_layers(model)

    it_sequence = iter(sequence_loader_ep) if sequence_loader_ep is not None else None
    it_still = iter(still_loader_ep) if still_loader_ep is not None else None

    steps = max(len(sequence_loader_ep), len(still_loader_ep))
    w_still = 1.0 - w_sequence

    loss_wsum = 0.0
    correct_wsum = 0.0
    wsum = 0.0

    pbar = tqdm(range(steps), desc=f"epoch {ep}/{epochs}", unit="step", colour=colour)

    for _ in pbar:
        got_sequence = False
        got_still = False

        loss_sequence = None
        loss_still = None
        logits_sequence = None
        logits_still = None
        y_sequence = None
        y_still = None

        if it_sequence is not None:
            try:
                xb, yb = next(it_sequence)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                logits_sequence = model.forward_sequence(xb)
                loss_sequence = criterion(logits_sequence, yb)

                got_sequence = True
                y_sequence = yb
            except StopIteration:
                it_sequence = None

        if it_still is not None:
            try:
                xs, ys = next(it_still)
                xs = xs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True).long()

                logits_still = model.forward_still(xs)
                loss_still = criterion(logits_still, ys)

                got_still = True
                y_still = ys
            except StopIteration:
                it_still = None

        if (not got_sequence) and (not got_still):
            continue

        if got_sequence and got_still:
            loss_mix = w_sequence * loss_sequence + w_still * loss_still
        elif got_sequence:
            loss_mix = loss_sequence
        else:
            loss_mix = loss_still

        optimizer.zero_grad(set_to_none=True)
        loss_mix.backward()
        optimizer.step()

        if got_sequence:
            bs = y_sequence.numel()
            loss_wsum += float(loss_sequence.item()) * (w_sequence * bs)
            correct_wsum += float((logits_sequence.argmax(1) == y_sequence).sum().item()) * w_sequence
            wsum += w_sequence * bs

        if got_still:
            bs = y_still.numel()
            loss_wsum += float(loss_still.item()) * (w_still * bs)
            correct_wsum += float((logits_still.argmax(1) == y_still).sum().item()) * w_still
            wsum += w_still * bs

        pbar.set_postfix({
            "training loss": f"{loss_wsum / max(1e-12, wsum):.4f}",
            "training accuracy": f"{correct_wsum / max(1e-12, wsum):.4f}",
        })

    train_time = time.perf_counter() - train_t0
    train_loss = loss_wsum / max(1e-12, wsum)
    train_accuracy = correct_wsum / max(1e-12, wsum)

    return train_loss, train_accuracy, train_time

# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def evaluate_hybrid(
    model,
    device,
    validation_loader,
    criterion,
    w_sequence=0.6,
):
    model.eval()

    v_loss_sum = 0.0
    v_correct_sum = 0.0
    v_n = 0
    w_still = 1.0 - w_sequence

    vbar = tqdm(
        validation_loader,
        desc="validation",
        unit="address",
        leave=False,
        colour="blue",
    )

    for batch in vbar:
        y = batch["y"]
        if torch.is_tensor(y):
            y = int(y.view(-1)[0].item())
        else:
            y = int(y)
        y_t = torch.tensor([y], device=device).long()

        have_sequence = False
        have_still = False
        logits_sequence = None
        logits_still = None

        sequence_runs = batch.get("sequence_runs", [])
        if sequence_runs is not None and len(sequence_runs) > 0:
            logits_runs = []
            run_lens = []

            for run in sequence_runs:
                run = run.to(device, non_blocking=True)

                # run: [T,3,H,W] -> [1,T,3,H,W]
                if run.dim() == 4:
                    run = run.unsqueeze(0)

                logits_runs.append(model.forward_sequence(run))   # [1,C]
                run_lens.append(run.shape[1])

            w = torch.tensor(run_lens, device=device, dtype=torch.float32)
            w = w / w.sum().clamp_min(1.0)

            logits_sequence = torch.stack(logits_runs, dim=0).squeeze(1)              # [R,C]
            logits_sequence = (logits_sequence * w.unsqueeze(1)).sum(dim=0, keepdim=True)  # [1,C]
            have_sequence = True

        still = batch.get("still", None)
        if still is not None:
            still = still.to(device, non_blocking=True)

            # still: [K,3,H,W] -> [1,K,3,H,W]
            if still.dim() == 4:
                still = still.unsqueeze(0)

            logits_still = model.forward_still(still)   # [1,C]
            have_still = True

        if have_sequence and have_still:
            logits = w_sequence * logits_sequence + w_still * logits_still
        elif have_sequence:
            logits = logits_sequence
        elif have_still:
            logits = logits_still
        else:
            continue

        loss = criterion(logits, y_t)

        v_loss_sum += float(loss.item())
        v_correct_sum += float((logits.argmax(1) == y_t).float().item())
        v_n += 1

        vbar.set_postfix({
            "Validation Loss": f"{v_loss_sum / max(1, v_n):.4f}",
            "Validation Accuracy": f"{v_correct_sum / max(1, v_n):.4f}",
            "Steps": v_n,
        })

    return v_loss_sum / max(1, v_n), v_correct_sum / max(1, v_n)
