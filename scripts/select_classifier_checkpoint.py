#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_classifier import build_model, compute_binary_metrics, normalize_dicom_to_tensor, pick_device
from training.checkpointing import load_checkpoint


class EvalDicomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, cache_dir: Path | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        y = torch.tensor(label, dtype=torch.long)
        item_idx = torch.tensor(idx, dtype=torch.long)
        return x, y, item_idx


def infer_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            logits = torch.nan_to_num(model(x), nan=0.0, posinf=30.0, neginf=-30.0)
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            probs.append(p)
            labels.append(y.numpy().astype(np.int64))
            indices.append(idx.numpy().astype(np.int64))

    y_prob = np.concatenate(probs, axis=0) if probs else np.array([])
    y_true = np.concatenate(labels, axis=0) if labels else np.array([])
    row_idx = np.concatenate(indices, axis=0) if indices else np.array([])
    return row_idx.reshape(-1), y_true.reshape(-1), y_prob.reshape(-1)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float,
    t_max: float,
    t_steps: int,
) -> pd.DataFrame:
    low = float(min(t_min, t_max))
    high = float(max(t_min, t_max))
    steps = int(max(t_steps, 2))
    rows: list[dict[str, float]] = []
    for thr in np.linspace(low, high, num=steps, dtype=np.float32):
        m = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
        rows.append({"threshold": float(thr), **{k: float(v) for k, v in m.items()}})
    return pd.DataFrame(rows)


def select_threshold(
    sweep_df: pd.DataFrame,
    mode: str,
    target_recall: float,
) -> tuple[float, dict[str, float], str]:
    if sweep_df.empty:
        raise RuntimeError("Threshold sweep is empty.")

    if mode == "max_f1":
        chosen = sweep_df.sort_values(["f1", "recall", "specificity", "auc"], ascending=False).iloc[0]
        reason = "max_f1_on_val"
    elif mode == "recall_floor":
        valid = sweep_df[sweep_df["recall"] >= target_recall]
        if not valid.empty:
            chosen = valid.sort_values(["specificity", "f1", "auc"], ascending=False).iloc[0]
            reason = f"max_specificity_with_recall_floor_{target_recall:.2f}"
        else:
            chosen = sweep_df.sort_values(["recall", "f1", "auc"], ascending=False).iloc[0]
            reason = f"fallback_max_recall_floor_not_met_{target_recall:.2f}"
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    threshold = float(chosen["threshold"])
    metrics = {
        "auc": float(chosen["auc"]),
        "accuracy": float(chosen["accuracy"]),
        "recall": float(chosen["recall"]),
        "specificity": float(chosen["specificity"]),
        "precision": float(chosen["precision"]),
        "f1": float(chosen["f1"]),
    }
    return threshold, metrics, reason


def attach_predictions(
    split_df: pd.DataFrame,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    meta = split_df.iloc[row_idx].copy().reset_index(drop=True)
    out = meta.copy()
    out["y_true"] = y_true.astype(int)
    out["prob_malignant"] = y_prob.astype(float)
    out["pred_label"] = (y_prob >= threshold).astype(int)
    out["threshold"] = float(threshold)
    return out


def error_tables(pred_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = pred_df[(pred_df["y_true"] == 0) & (pred_df["pred_label"] == 1)].copy()
    fn = pred_df[(pred_df["y_true"] == 1) & (pred_df["pred_label"] == 0)].copy()
    fp = fp.sort_values("prob_malignant", ascending=False).head(top_k).reset_index(drop=True)
    fn = fn.sort_values("prob_malignant", ascending=True).head(top_k).reset_index(drop=True)
    return fp, fn


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    manifest: pd.DataFrame,
    model_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold_mode: str,
    target_recall: float,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    cache_dir: Path | None,
    top_k_errors: int,
) -> dict[str, Any]:
    val_df = manifest[manifest["split"] == "val"].copy().reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    if val_df.empty or test_df.empty:
        raise RuntimeError("Manifest must contain non-empty val and test splits.")

    val_cache = cache_dir / "val" if cache_dir else None
    test_cache = cache_dir / "test" if cache_dir else None
    val_ds = EvalDicomDataset(val_df, image_size=image_size, cache_dir=val_cache)
    test_ds = EvalDicomDataset(test_df, image_size=image_size, cache_dir=test_cache)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(model_name, pretrained=False).to(device)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    val_idx, val_y_true, val_y_prob = infer_split(model=model, loader=val_loader, device=device)
    test_idx, test_y_true, test_y_prob = infer_split(model=model, loader=test_loader, device=device)

    val_sweep = threshold_sweep(
        y_true=val_y_true,
        y_prob=val_y_prob,
        t_min=threshold_min,
        t_max=threshold_max,
        t_steps=threshold_steps,
    )
    chosen_thr, val_metrics_chosen, reason = select_threshold(
        sweep_df=val_sweep,
        mode=threshold_mode,
        target_recall=target_recall,
    )
    test_metrics_chosen = compute_binary_metrics(y_true=test_y_true, y_prob=test_y_prob, threshold=chosen_thr)

    val_pred_df = attach_predictions(val_df, val_idx, val_y_true, val_y_prob, chosen_thr)
    test_pred_df = attach_predictions(test_df, test_idx, test_y_true, test_y_prob, chosen_thr)
    fp_df, fn_df = error_tables(test_pred_df, top_k=top_k_errors)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "selected_threshold": float(chosen_thr),
        "selection_reason": reason,
        "val_metrics_at_selected_threshold": {k: float(v) for k, v in val_metrics_chosen.items()},
        "test_metrics_at_selected_threshold": {k: float(v) for k, v in test_metrics_chosen.items()},
        "val_sweep_df": val_sweep,
        "val_predictions_df": val_pred_df,
        "test_predictions_df": test_pred_df,
        "false_positives_df": fp_df,
        "false_negatives_df": fn_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select best checkpoint + threshold for classifier run")
    parser.add_argument("--manifest", required=True, help="CSV with dcm_path,label,split")
    parser.add_argument("--run-dir", required=True, help="Run directory with checkpoints")
    parser.add_argument(
        "--model",
        required=True,
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "resnet18"],
    )
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--threshold-mode", default="recall_floor", choices=["recall_floor", "max_f1"])
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--top-k-errors", type=int, default=25)
    parser.add_argument(
        "--checkpoint-names",
        default="last.pt,best_auc.pt,best_recall.pt",
        help="Comma-separated checkpoint filenames from run checkpoints directory.",
    )
    parser.add_argument(
        "--include-epoch-checkpoints",
        action="store_true",
        help="Also evaluate all epoch_*.pt checkpoints found in run/checkpoints.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="If >0, keep at most this many checkpoints (priority to named checkpoints + latest epochs).",
    )
    parser.add_argument("--cache-dir", default="", help="Optional cache for eval tensors")
    parser.add_argument("--out-dir", default="", help="Output directory (default: <run-dir>/analysis/checkpoint_selection)")
    args = parser.parse_args()

    target_recall = float(min(max(args.target_recall, 0.0), 1.0))
    threshold_steps = max(int(args.threshold_steps), 2)
    threshold_min = float(min(max(args.threshold_min, 0.0), 1.0))
    threshold_max = float(min(max(args.threshold_max, 0.0), 1.0))

    manifest_path = Path(args.manifest).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "analysis" / "checkpoint_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    required_cols = {"dcm_path", "label", "split"}
    if not required_cols.issubset(manifest.columns):
        missing = sorted(required_cols - set(manifest.columns))
        raise ValueError(f"Manifest missing columns: {missing}")

    device = pick_device(args.device)
    checkpoint_names = [x.strip() for x in args.checkpoint_names.split(",") if x.strip()]
    checkpoint_paths = [ckpt_dir / name for name in checkpoint_names if (ckpt_dir / name).exists()]

    if args.include_epoch_checkpoints:
        epoch_ckpts = sorted(
            ckpt_dir.glob("epoch_*.pt"),
            key=lambda p: int(str(p.stem).split("_")[-1]) if str(p.stem).split("_")[-1].isdigit() else -1,
        )
        checkpoint_paths.extend(epoch_ckpts)

    checkpoint_paths = list(dict.fromkeys([p.resolve() for p in checkpoint_paths]))

    if args.max_checkpoints > 0 and len(checkpoint_paths) > args.max_checkpoints:
        named_abs = [p.resolve() for p in (ckpt_dir / name for name in checkpoint_names) if p.exists()]
        named_abs = list(dict.fromkeys(named_abs))
        remaining = [p for p in checkpoint_paths if p not in named_abs]
        # Keep newest epoch checkpoints if trimming is needed.
        remaining = sorted(
            remaining,
            key=lambda p: int(str(p.stem).split("_")[-1]) if str(p.stem).split("_")[-1].isdigit() else -1,
            reverse=True,
        )
        limit_left = max(args.max_checkpoints - len(named_abs), 0)
        checkpoint_paths = named_abs + remaining[:limit_left]

    if not checkpoint_paths:
        raise RuntimeError(f"No checkpoints found for names={checkpoint_names} in {ckpt_dir}")

    results: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    for ckpt_path in checkpoint_paths:
        result = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            manifest=manifest,
            model_name=args.model,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            threshold_mode=args.threshold_mode,
            target_recall=target_recall,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
            cache_dir=cache_dir,
            top_k_errors=args.top_k_errors,
        )
        results.append(result)

        vm = result["val_metrics_at_selected_threshold"]
        tm = result["test_metrics_at_selected_threshold"]
        ranking_rows.append(
            {
                "checkpoint_name": result["checkpoint_name"],
                "checkpoint_path": result["checkpoint_path"],
                "selected_threshold": result["selected_threshold"],
                "selection_reason": result["selection_reason"],
                "val_meets_target_recall": 1 if vm["recall"] >= target_recall else 0,
                "val_auc": vm["auc"],
                "val_recall": vm["recall"],
                "val_specificity": vm["specificity"],
                "val_f1": vm["f1"],
                "test_auc": tm["auc"],
                "test_recall": tm["recall"],
                "test_specificity": tm["specificity"],
                "test_f1": tm["f1"],
                "test_accuracy": tm["accuracy"],
            }
        )

        base = ckpt_path.stem
        result["val_sweep_df"].to_csv(out_dir / f"{base}_val_threshold_sweep.csv", index=False)
        result["val_predictions_df"].to_csv(out_dir / f"{base}_val_predictions.csv", index=False)
        result["test_predictions_df"].to_csv(out_dir / f"{base}_test_predictions.csv", index=False)
        result["false_positives_df"].to_csv(out_dir / f"{base}_false_positives_top{args.top_k_errors}.csv", index=False)
        result["false_negatives_df"].to_csv(out_dir / f"{base}_false_negatives_top{args.top_k_errors}.csv", index=False)

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(
        ["val_meets_target_recall", "val_specificity", "val_f1", "val_auc"],
        ascending=False,
    ).reset_index(drop=True)
    ranking_path = out_dir / "checkpoint_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    best_row = ranking_df.iloc[0].to_dict()
    champion = next(x for x in results if x["checkpoint_name"] == best_row["checkpoint_name"])

    summary = {
        "manifest": str(manifest_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "threshold_mode": args.threshold_mode,
        "target_recall": target_recall,
        "threshold_range": [threshold_min, threshold_max],
        "threshold_steps": threshold_steps,
        "evaluated_checkpoints": checkpoint_names,
        "ranking_csv": str(ranking_path),
        "champion": {
            "checkpoint_name": champion["checkpoint_name"],
            "checkpoint_path": champion["checkpoint_path"],
            "selected_threshold": champion["selected_threshold"],
            "selection_reason": champion["selection_reason"],
            "val_metrics_at_selected_threshold": champion["val_metrics_at_selected_threshold"],
            "test_metrics_at_selected_threshold": champion["test_metrics_at_selected_threshold"],
            "false_positive_count": int(len(champion["false_positives_df"])),
            "false_negative_count": int(len(champion["false_negatives_df"])),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
