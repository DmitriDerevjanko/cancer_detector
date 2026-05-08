#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TVF

from training.checkpointing import (
    CheckpointBundle,
    append_metrics_row,
    ensure_run_dirs,
    load_checkpoint,
    save_checkpoint,
    save_run_state,
)


def format_seconds_hhmmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(explicit: str) -> torch.device:
    if explicit != "auto":
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_dicom_to_tensor(dcm_path: Path, image_size: int) -> torch.Tensor:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    t = t.repeat(3, 1, 1)
    return t.clamp(0.0, 1.0)


def apply_train_aug(x: torch.Tensor, aug_profile: str) -> torch.Tensor:
    profile = aug_profile.lower().strip()
    if profile == "none":
        return x.clamp(0.0, 1.0)

    h, w = int(x.shape[1]), int(x.shape[2])

    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])

    if profile == "strong" and torch.rand(1).item() < 0.35:
        i, j, hh, ww = RandomResizedCrop.get_params(
            x,
            scale=(0.82, 1.0),
            ratio=(0.9, 1.1),
        )
        x = TVF.resized_crop(
            x,
            i,
            j,
            hh,
            ww,
            size=[h, w],
            interpolation=InterpolationMode.BILINEAR,
        )

    if profile == "strong" and torch.rand(1).item() < 0.55:
        angle = float((torch.rand(1).item() * 2.0 - 1.0) * 8.0)
        tx = int((torch.rand(1).item() * 2.0 - 1.0) * 0.04 * w)
        ty = int((torch.rand(1).item() * 2.0 - 1.0) * 0.04 * h)
        scale = float(1.0 + (torch.rand(1).item() * 2.0 - 1.0) * 0.10)
        shear = float((torch.rand(1).item() * 2.0 - 1.0) * 3.0)
        x = TVF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[shear, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    if profile == "strong":
        contrast = 0.8 + 0.45 * torch.rand(1).item()
        brightness = -0.10 + 0.20 * torch.rand(1).item()
    else:
        contrast = 0.9 + 0.2 * torch.rand(1).item()
        brightness = -0.05 + 0.10 * torch.rand(1).item()
    x = x * contrast + brightness

    if profile == "strong" and torch.rand(1).item() < 0.35:
        gamma = 0.8 + 0.4 * torch.rand(1).item()
        x = torch.pow(torch.clamp(x, min=1e-6), gamma)

    if profile == "strong" and torch.rand(1).item() < 0.25:
        x = TVF.gaussian_blur(x, kernel_size=3, sigma=(0.2, 1.2))

    noise_std = (0.03 if profile == "strong" else 0.01) * torch.rand(1).item()
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    if profile == "strong" and torch.rand(1).item() < 0.20:
        cut_h = max(1, int(h * (0.08 + 0.10 * torch.rand(1).item())))
        cut_w = max(1, int(w * (0.08 + 0.10 * torch.rand(1).item())))
        top = int(torch.randint(0, max(h - cut_h + 1, 1), (1,)).item())
        left = int(torch.randint(0, max(w - cut_w + 1, 1), (1,)).item())
        fill_v = float(torch.mean(x))
        x[:, top : top + cut_h, left : left + cut_w] = fill_v

    return x.clamp(0.0, 1.0)


class DicomBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        augment: bool,
        aug_profile: str = "basic",
        cache_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.aug_profile = aug_profile.lower().strip()
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, dcm_path: Path) -> Path:
        key = f"{dcm_path.resolve()}::{self.image_size}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        assert self.cache_dir is not None
        return self.cache_dir / f"{digest}.pt"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        dcm_path = Path(str(row["dcm_path"])).expanduser().resolve()
        label = 1 if str(row["label"]).strip().lower() == "malignant" else 0

        if self.cache_dir is not None:
            cache_file = self._cache_path(dcm_path)
            if cache_file.exists():
                x = torch.load(cache_file, map_location="cpu", weights_only=False)
            else:
                x = normalize_dicom_to_tensor(dcm_path=dcm_path, image_size=self.image_size)
                torch.save(x, cache_file)
        else:
            x = normalize_dicom_to_tensor(dcm_path=dcm_path, image_size=self.image_size)

        if self.augment:
            x = apply_train_aug(x=x, aug_profile=self.aug_profile)

        y = torch.tensor(label, dtype=torch.long)
        return x, y


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    name = model_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model

    if name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model

    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(max(gamma, 0.0))
        self.weight = weight
        self.label_smoothing = float(max(label_smoothing, 0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        if self.gamma <= 0.0:
            return ce.mean()

        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(dim=1, index=target.view(-1, 1)).squeeze(1).clamp_(1e-6, 1.0)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        return (focal_factor * ce).mean()


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    name = model_name.lower().strip()
    if name == "resnet18":
        for param_name, param in model.named_parameters():
            if param_name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = trainable
        return

    if name in {"efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s"}:
        for param in model.features.parameters():
            param.requires_grad = trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unsupported model for freezing: {model_name}")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim > 1:
        if y_true.shape[-1] == 1:
            y_true = y_true.reshape(-1)
        else:
            y_true = y_true[:, -1]

    if y_prob.ndim > 1:
        if y_prob.shape[-1] == 1:
            y_prob = y_prob.reshape(-1)
        else:
            y_prob = y_prob[:, -1]

    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    if y_true.size == 0:
        return {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true_i = y_true.astype(np.int64)

    tp = int(((y_pred == 1) & (y_true_i == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_i == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_i == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_i == 1)).sum())

    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    if len(np.unique(y_true_i)) < 2:
        auc = float("nan")
        warnings.warn(
            f"AUC undefined: only one class in y_true (unique={np.unique(y_true_i).tolist()})",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        try:
            auc = float(roc_auc_score(y_true_i, y_prob))
        except ValueError as exc:
            auc = float("nan")
            warnings.warn(
                "AUC computation failed: "
                f"{exc}. y_true_shape={y_true_i.shape}, y_prob_shape={y_prob.shape}, "
                f"y_true_unique={np.unique(y_true_i).tolist()}, "
                f"y_prob_min={float(np.min(y_prob)) if y_prob.size else 'na'}, "
                f"y_prob_max={float(np.max(y_prob)) if y_prob.size else 'na'}",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "auc": auc,
        "accuracy": float(accuracy),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def pick_threshold_from_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    mode: str,
    fixed_threshold: float,
    target_recall: float,
    t_min: float,
    t_max: float,
    t_steps: int,
) -> tuple[float, dict[str, float]]:
    if mode == "fixed":
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=fixed_threshold)
        return float(fixed_threshold), metrics

    low = float(min(t_min, t_max))
    high = float(max(t_min, t_max))
    steps = int(max(t_steps, 2))
    grid = np.linspace(low, high, num=steps, dtype=np.float32)

    best_thr = float(fixed_threshold)
    best_metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=best_thr)

    if mode == "val_f1":
        best_key = (-1.0, -1.0, -1.0)  # f1, recall, specificity
        for thr in grid:
            m = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
            key = (
                float(np.nan_to_num(m["f1"], nan=-1.0)),
                float(np.nan_to_num(m["recall"], nan=-1.0)),
                float(np.nan_to_num(m["specificity"], nan=-1.0)),
            )
            if key > best_key:
                best_key = key
                best_thr = float(thr)
                best_metrics = m
        return best_thr, best_metrics

    if mode == "val_recall_floor":
        valid: list[tuple[float, dict[str, float]]] = []
        fallback: list[tuple[float, dict[str, float]]] = []
        for thr in grid:
            m = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
            rec = float(np.nan_to_num(m["recall"], nan=-1.0))
            fallback.append((float(thr), m))
            if rec >= float(target_recall):
                valid.append((float(thr), m))

        if valid:
            best_thr, best_metrics = max(
                valid,
                key=lambda item: (
                    float(np.nan_to_num(item[1]["specificity"], nan=-1.0)),
                    float(np.nan_to_num(item[1]["f1"], nan=-1.0)),
                    float(np.nan_to_num(item[1]["auc"], nan=-1.0)),
                ),
            )
            return float(best_thr), best_metrics

        best_thr, best_metrics = max(
            fallback,
            key=lambda item: (
                float(np.nan_to_num(item[1]["recall"], nan=-1.0)),
                float(np.nan_to_num(item[1]["f1"], nan=-1.0)),
                float(np.nan_to_num(item[1]["auc"], nan=-1.0)),
            ),
        )
        return float(best_thr), best_metrics

    raise ValueError(f"Unsupported threshold mode: {mode}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    mixup_alpha: float,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total = 0
    steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha > 0.0 and x.shape[0] > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam = max(lam, 1.0 - lam)
            perm = torch.randperm(x.shape[0], device=device)
            x_mix = lam * x + (1.0 - lam) * x[perm]
            y_perm = y[perm]

            logits = torch.nan_to_num(model(x_mix), nan=0.0, posinf=30.0, neginf=-30.0)
            loss_a = criterion(logits, y)
            loss_b = criterion(logits, y_perm)
            loss = lam * loss_a + (1.0 - lam) * loss_b
        else:
            logits = torch.nan_to_num(model(x), nan=0.0, posinf=30.0, neginf=-30.0)
            loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_n = y.shape[0]
        total_loss += loss.item() * batch_n
        total += batch_n
        steps += 1

    return total_loss / max(total, 1), steps


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    return_outputs: bool = False,
) -> tuple[float, dict[str, float], np.ndarray | None, np.ndarray | None]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            y_cpu = y.numpy().astype(np.int64)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True, dtype=torch.long)

            logits = torch.nan_to_num(model(x), nan=0.0, posinf=30.0, neginf=-30.0)
            loss = criterion(logits, y)

            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            l = y_cpu

            probs.append(p)
            labels.append(l)

            batch_n = y.shape[0]
            total_loss += loss.item() * batch_n
            total += batch_n

    y_prob = np.concatenate(probs, axis=0) if probs else np.array([])
    y_true = np.concatenate(labels, axis=0) if labels else np.array([])
    y_prob = np.asarray(y_prob).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    val_loss = total_loss / max(total, 1)
    if return_outputs:
        return val_loss, metrics, y_true, y_prob
    return val_loss, metrics, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train binary classifier with robust checkpoint/resume")
    parser.add_argument("--manifest", required=True, help="CSV with dcm_path,label,split")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "resnet18"],
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"], help="Classification loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Used when --loss=focal")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="0 disables mixup")
    parser.add_argument(
        "--aug-profile",
        default="basic",
        choices=["none", "basic", "strong"],
        help="Train-time augmentation profile",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze feature backbone for N initial epochs to stabilize head training",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        default="fixed",
        choices=["fixed", "val_f1", "val_recall_floor"],
        help="How to select decision threshold each epoch.",
    )
    parser.add_argument("--target-recall", type=float, default=0.85, help="Used for threshold-mode=val_recall_floor")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="", help="Initialize model weights from checkpoint path")
    parser.add_argument("--cache-dir", default="", help="Optional cache directory for preprocessed tensors")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--balanced-sampler", action="store_true", help="Use weighted sampling for train split")
    parser.add_argument("--disable-class-weights", action="store_true", help="Disable class-weighted loss")
    args = parser.parse_args()

    args.target_recall = float(min(max(args.target_recall, 0.0), 1.0))
    args.label_smoothing = float(min(max(args.label_smoothing, 0.0), 0.2))
    args.mixup_alpha = float(max(args.mixup_alpha, 0.0))
    args.focal_gamma = float(max(args.focal_gamma, 0.0))
    args.freeze_backbone_epochs = int(max(args.freeze_backbone_epochs, 0))
    if args.threshold_steps < 2:
        args.threshold_steps = 2
    args.threshold_min = float(min(max(args.threshold_min, 0.0), 1.0))
    args.threshold_max = float(min(max(args.threshold_max, 0.0), 1.0))

    seed_everything(args.seed)
    device = pick_device(args.device)

    run_dir, ckpt_dir, logs_dir = ensure_run_dirs(Path(args.run_dir).expanduser().resolve())
    cache_root = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    manifest = pd.read_csv(Path(args.manifest).expanduser().resolve())
    required_cols = {"dcm_path", "label", "split"}
    if not required_cols.issubset(manifest.columns):
        missing = sorted(required_cols - set(manifest.columns))
        raise ValueError(f"Manifest missing columns: {missing}")

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    if args.max_train_samples > 0:
        train_df = train_df.sample(min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if args.max_val_samples > 0:
        val_df = val_df.sample(min(args.max_val_samples, len(val_df)), random_state=args.seed)
    if args.max_test_samples > 0:
        test_df = test_df.sample(min(args.max_test_samples, len(test_df)), random_state=args.seed)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Expected non-empty train/val/test splits.")

    train_cache = cache_root / "train" if cache_root else None
    val_cache = cache_root / "val" if cache_root else None
    test_cache = cache_root / "test" if cache_root else None

    train_ds = DicomBinaryDataset(
        train_df,
        image_size=args.image_size,
        augment=True,
        aug_profile=args.aug_profile,
        cache_dir=train_cache,
    )
    val_ds = DicomBinaryDataset(
        val_df,
        image_size=args.image_size,
        augment=False,
        aug_profile="none",
        cache_dir=val_cache,
    )
    test_ds = DicomBinaryDataset(
        test_df,
        image_size=args.image_size,
        augment=False,
        aug_profile="none",
        cache_dir=test_cache,
    )

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        label_ids = (
            train_ds.df["label"].astype(str).str.strip().str.lower().map({"benign": 0, "malignant": 1}).fillna(0).astype(int).to_numpy()
        )
        class_counts = np.bincount(label_ids, minlength=2).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[label_ids]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    model = build_model(args.model, pretrained=args.pretrained).to(device)

    pos = float((train_df["label"].str.lower() == "malignant").sum())
    neg = float(len(train_df) - pos)
    total = max(pos + neg, 1.0)
    class_weights = torch.tensor(
        [total / max(2.0 * neg, 1.0), total / max(2.0 * pos, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    loss_weights = None if args.disable_class_weights else class_weights
    if args.loss == "focal":
        criterion = FocalCrossEntropyLoss(
            gamma=args.focal_gamma,
            weight=loss_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=args.label_smoothing,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    print(
        f"train_samples={len(train_df)} val_samples={len(val_df)} test_samples={len(test_df)} "
        f"pos={int(pos)} neg={int(neg)} class_weights={'off' if args.disable_class_weights else 'on'} "
        f"balanced_sampler={'on' if args.balanced_sampler else 'off'} "
        f"loss={args.loss} gamma={args.focal_gamma:.2f} "
        f"label_smoothing={args.label_smoothing:.3f} mixup={args.mixup_alpha:.3f} "
        f"aug={args.aug_profile} freeze_backbone_epochs={args.freeze_backbone_epochs}"
    , flush=True)

    start_epoch = 0
    global_step = 0
    best_auc = -math.inf
    best_recall = -math.inf
    best_auc_epoch = -1
    best_recall_epoch = -1
    current_threshold = float(args.threshold)
    best_auc_threshold = float(args.threshold)
    best_recall_threshold = float(args.threshold)
    no_improve_epochs = 0

    last_ckpt = ckpt_dir / "last.pt"

    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both.")

    if args.init_checkpoint and not args.resume:
        init_ckpt_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt_path}")
        init_ckpt = load_checkpoint(init_ckpt_path, map_location="cpu")
        model.load_state_dict(init_ckpt["model_state_dict"])
        init_metrics = init_ckpt.get("metrics", {})
        if "threshold" in init_metrics:
            current_threshold = float(init_metrics["threshold"])
            best_auc_threshold = float(init_metrics.get("best_auc_threshold", current_threshold))
            best_recall_threshold = float(init_metrics.get("best_recall_threshold", current_threshold))
        print(f"Initialized model weights from checkpoint: {init_ckpt_path}")

    if args.resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        m = ckpt.get("metrics", {})
        best_auc = float(m.get("best_auc", -math.inf))
        best_recall = float(m.get("best_recall", -math.inf))
        best_auc_epoch = int(m.get("best_auc_epoch", -1))
        best_recall_epoch = int(m.get("best_recall_epoch", -1))
        current_threshold = float(m.get("threshold", current_threshold))
        best_auc_threshold = float(m.get("best_auc_threshold", best_auc_threshold))
        best_recall_threshold = float(m.get("best_recall_threshold", best_recall_threshold))

        print(
            f"Resumed from epoch={start_epoch}, best_auc={best_auc:.4f} "
            f"(epoch={best_auc_epoch}), best_recall={best_recall:.4f} (epoch={best_recall_epoch}), "
            f"threshold={current_threshold:.3f}"
        , flush=True)

    interrupted = {"flag": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        interrupted["flag"] = True
        print(f"\\nSignal {signum} received: will stop after current epoch and keep checkpoint.", flush=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_start = time.time()
    backbone_frozen = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        last_epoch = epoch

        should_freeze = epoch < args.freeze_backbone_epochs
        if should_freeze != backbone_frozen:
            set_backbone_trainable(model=model, model_name=args.model, trainable=(not should_freeze))
            backbone_frozen = should_freeze
            print(f"epoch={epoch:03d} backbone={'frozen' if should_freeze else 'unfrozen'}", flush=True)

        train_loss, steps = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            mixup_alpha=args.mixup_alpha,
        )
        global_step += steps

        val_loss, val_metrics, y_true_val, y_prob_val = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=current_threshold,
            return_outputs=True,
        )

        if args.threshold_mode != "fixed" and y_true_val is not None and y_prob_val is not None:
            current_threshold, val_metrics = pick_threshold_from_val(
                y_true=y_true_val,
                y_prob=y_prob_val,
                mode=args.threshold_mode,
                fixed_threshold=current_threshold,
                target_recall=args.target_recall,
                t_min=args.threshold_min,
                t_max=args.threshold_max,
                t_steps=args.threshold_steps,
            )

        scheduler.step()

        val_auc = float(val_metrics["auc"])
        val_recall = float(val_metrics["recall"])

        auc_improved = (not math.isnan(val_auc)) and val_auc > best_auc
        recall_improved = (not math.isnan(val_recall)) and (
            val_recall > best_recall
            or (math.isclose(val_recall, best_recall, rel_tol=1e-9, abs_tol=1e-12) and val_auc > best_auc)
        )

        if auc_improved:
            best_auc = val_auc
            best_auc_epoch = epoch
            best_auc_threshold = current_threshold
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if recall_improved:
            best_recall = val_recall
            best_recall_epoch = epoch
            best_recall_threshold = current_threshold

        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_recall": float(val_recall),
            "val_specificity": float(val_metrics["specificity"]),
            "val_precision": float(val_metrics["precision"]),
            "val_f1": float(val_metrics["f1"]),
            "threshold": float(current_threshold),
            "threshold_mode": str(args.threshold_mode),
            "target_recall": float(args.target_recall),
            "loss_name": str(args.loss),
            "focal_gamma": float(args.focal_gamma),
            "label_smoothing": float(args.label_smoothing),
            "mixup_alpha": float(args.mixup_alpha),
            "aug_profile": str(args.aug_profile),
            "backbone_frozen": bool(backbone_frozen),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "best_auc": float(best_auc),
            "best_recall": float(best_recall),
            "best_auc_epoch": int(best_auc_epoch),
            "best_recall_epoch": int(best_recall_epoch),
            "best_auc_threshold": float(best_auc_threshold),
            "best_recall_threshold": float(best_recall_threshold),
            "epoch_time_sec": float(time.time() - epoch_start),
            "elapsed_sec": float(time.time() - run_start),
        }

        bundle = CheckpointBundle(
            epoch=epoch,
            global_step=global_step,
            best_metric=best_auc,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
        )

        save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name="last.pt")

        if auc_improved:
            save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name="best_auc.pt")

        if recall_improved:
            save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name="best_recall.pt")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_dir=ckpt_dir, bundle=bundle, metrics=metrics, file_name=f"epoch_{epoch:03d}.pt")

        append_metrics_row(
            logs_dir=logs_dir,
            row={
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
            },
        )

        save_run_state(
            run_dir=run_dir,
            payload={
                "status": "running",
                "epoch": epoch,
                "global_step": global_step,
                "best_auc": best_auc,
                "best_recall": best_recall,
                "best_auc_epoch": best_auc_epoch,
                "best_recall_epoch": best_recall_epoch,
                "threshold": current_threshold,
                "threshold_mode": args.threshold_mode,
                "best_auc_threshold": best_auc_threshold,
                "best_recall_threshold": best_recall_threshold,
                "checkpoint_last": str(ckpt_dir / "last.pt"),
                "checkpoint_best_auc": str(ckpt_dir / "best_auc.pt"),
                "checkpoint_best_recall": str(ckpt_dir / "best_recall.pt"),
                "latest_metrics": metrics,
            },
        )

        done_epochs = epoch + 1
        left_epochs = max(args.epochs - done_epochs, 0)
        avg_epoch_sec = float(time.time() - run_start) / max(done_epochs - start_epoch, 1)
        eta_sec = avg_epoch_sec * left_epochs
        progress = f"{done_epochs:03d}/{args.epochs:03d}"

        print(
            f"epoch={epoch:03d} [{progress}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"auc={val_auc:.4f} recall={val_recall:.4f} spec={val_metrics['specificity']:.4f} "
            f"thr={current_threshold:.3f} best_auc={best_auc:.4f} best_recall={best_recall:.4f} "
            f"epoch_time={format_seconds_hhmmss(metrics['epoch_time_sec'])} "
            f"eta={format_seconds_hhmmss(eta_sec)}"
            ,
            flush=True,
        )

        if interrupted["flag"]:
            save_run_state(
                run_dir=run_dir,
                payload={
                    "status": "interrupted",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_auc": best_auc,
                    "best_recall": best_recall,
                    "best_auc_epoch": best_auc_epoch,
                    "best_recall_epoch": best_recall_epoch,
                    "threshold": current_threshold,
                    "threshold_mode": args.threshold_mode,
                    "best_auc_threshold": best_auc_threshold,
                    "best_recall_threshold": best_recall_threshold,
                    "checkpoint_last": str(ckpt_dir / "last.pt"),
                    "checkpoint_best_auc": str(ckpt_dir / "best_auc.pt"),
                    "checkpoint_best_recall": str(ckpt_dir / "best_recall.pt"),
                    "latest_metrics": metrics,
                },
            )
            print("Stopped safely. Resume with --resume", flush=True)
            return 0

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch={epoch} (no val_auc improvement for {no_improve_epochs} epochs).",
                flush=True,
            )
            break

    test_loss_last, test_metrics_last, _, _ = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=current_threshold,
        return_outputs=False,
    )

    best_auc_test: dict[str, Any] | None = None
    best_auc_ckpt = ckpt_dir / "best_auc.pt"
    if best_auc_ckpt.exists():
        ckpt_best_auc = load_checkpoint(best_auc_ckpt, map_location="cpu")
        model.load_state_dict(ckpt_best_auc["model_state_dict"])
        thr_best_auc = float(ckpt_best_auc.get("metrics", {}).get("threshold", best_auc_threshold))
        loss_ba, metrics_ba, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=thr_best_auc,
            return_outputs=False,
        )
        best_auc_test = {
            "checkpoint": str(best_auc_ckpt),
            "threshold": float(thr_best_auc),
            "loss": float(loss_ba),
            **{k: float(v) for k, v in metrics_ba.items()},
        }

    best_recall_test: dict[str, Any] | None = None
    best_recall_ckpt = ckpt_dir / "best_recall.pt"
    if best_recall_ckpt.exists():
        ckpt_best_recall = load_checkpoint(best_recall_ckpt, map_location="cpu")
        model.load_state_dict(ckpt_best_recall["model_state_dict"])
        thr_best_recall = float(ckpt_best_recall.get("metrics", {}).get("threshold", best_recall_threshold))
        loss_br, metrics_br, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=thr_best_recall,
            return_outputs=False,
        )
        best_recall_test = {
            "checkpoint": str(best_recall_ckpt),
            "threshold": float(thr_best_recall),
            "loss": float(loss_br),
            **{k: float(v) for k, v in metrics_br.items()},
        }

    final_payload = {
        "status": "completed",
        "epoch": last_epoch,
        "global_step": global_step,
        "best_auc": best_auc,
        "best_recall": best_recall,
        "best_auc_epoch": best_auc_epoch,
        "best_recall_epoch": best_recall_epoch,
        "checkpoint_last": str(ckpt_dir / "last.pt"),
        "checkpoint_best_auc": str(ckpt_dir / "best_auc.pt"),
        "checkpoint_best_recall": str(ckpt_dir / "best_recall.pt"),
        "threshold_mode": str(args.threshold_mode),
        "target_recall": float(args.target_recall),
        "threshold_final": float(current_threshold),
        "threshold_best_auc": float(best_auc_threshold),
        "threshold_best_recall": float(best_recall_threshold),
        "test_last": {
            "threshold": float(current_threshold),
            "loss": float(test_loss_last),
            **{k: float(v) for k, v in test_metrics_last.items()},
        },
        "test_best_auc": best_auc_test,
        "test_best_recall": best_recall_test,
    }
    save_run_state(run_dir=run_dir, payload=final_payload)

    print(
        "test(last) "
        f"loss={test_loss_last:.4f} auc={test_metrics_last['auc']:.4f} recall={test_metrics_last['recall']:.4f} "
        f"spec={test_metrics_last['specificity']:.4f} acc={test_metrics_last['accuracy']:.4f} "
        f"thr={current_threshold:.3f}"
    , flush=True)
    if best_auc_test is not None:
        print(
            "test(best_auc) "
            f"loss={best_auc_test['loss']:.4f} auc={best_auc_test['auc']:.4f} "
            f"recall={best_auc_test['recall']:.4f} spec={best_auc_test['specificity']:.4f} "
            f"acc={best_auc_test['accuracy']:.4f} thr={best_auc_test['threshold']:.3f}"
        , flush=True)
    if best_recall_test is not None:
        print(
            "test(best_recall) "
            f"loss={best_recall_test['loss']:.4f} auc={best_recall_test['auc']:.4f} "
            f"recall={best_recall_test['recall']:.4f} spec={best_recall_test['specificity']:.4f} "
            f"acc={best_recall_test['accuracy']:.4f} thr={best_recall_test['threshold']:.3f}"
        , flush=True)
    print("Training completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
