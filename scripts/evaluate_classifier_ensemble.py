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


def pick_threshold(sweep_df: pd.DataFrame, mode: str, target_recall: float) -> tuple[float, str]:
    if mode == "max_f1":
        chosen = sweep_df.sort_values(["f1", "recall", "specificity", "auc"], ascending=False).iloc[0]
        return float(chosen["threshold"]), "max_f1_on_val"

    valid = sweep_df[sweep_df["recall"] >= target_recall]
    if not valid.empty:
        chosen = valid.sort_values(["specificity", "f1", "auc"], ascending=False).iloc[0]
        return float(chosen["threshold"]), f"max_specificity_with_recall_floor_{target_recall:.2f}"

    chosen = sweep_df.sort_values(["recall", "f1", "auc"], ascending=False).iloc[0]
    return float(chosen["threshold"]), f"fallback_max_recall_floor_not_met_{target_recall:.2f}"


def infer_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_hflip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)

            logits = torch.nan_to_num(model(x), nan=0.0, posinf=30.0, neginf=-30.0)
            p = torch.softmax(logits, dim=1)[:, 1]

            if tta_hflip:
                x_flip = torch.flip(x, dims=[3])
                logits_flip = torch.nan_to_num(model(x_flip), nan=0.0, posinf=30.0, neginf=-30.0)
                p_flip = torch.softmax(logits_flip, dim=1)[:, 1]
                p = 0.5 * (p + p_flip)

            probs.append(p.detach().cpu().numpy())
            labels.append(y.numpy().astype(np.int64))
            indices.append(idx.numpy().astype(np.int64))

    y_prob = np.concatenate(probs, axis=0) if probs else np.array([])
    y_true = np.concatenate(labels, axis=0) if labels else np.array([])
    row_idx = np.concatenate(indices, axis=0) if indices else np.array([])
    return row_idx.reshape(-1), y_true.reshape(-1), y_prob.reshape(-1)


def eval_split_ensemble(
    loader: DataLoader,
    models: list[nn.Module],
    weights: np.ndarray,
    device: torch.device,
    tta_hflip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_ref: np.ndarray | None = None
    y_ref: np.ndarray | None = None
    prob_accum: np.ndarray | None = None

    for model, w in zip(models, weights, strict=True):
        idx, y_true, y_prob = infer_probs(model=model, loader=loader, device=device, tta_hflip=tta_hflip)

        if idx_ref is None:
            idx_ref = idx
            y_ref = y_true
            prob_accum = w * y_prob.astype(np.float64)
        else:
            if not np.array_equal(idx_ref, idx):
                raise RuntimeError("Prediction index order mismatch across models.")
            if not np.array_equal(y_ref, y_true):
                raise RuntimeError("Label mismatch across models.")
            assert prob_accum is not None
            prob_accum += w * y_prob.astype(np.float64)

    assert idx_ref is not None and y_ref is not None and prob_accum is not None
    return idx_ref, y_ref, prob_accum.astype(np.float32)


def attach_predictions(split_df: pd.DataFrame, row_idx: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    out = split_df.iloc[row_idx].copy().reset_index(drop=True)
    out["y_true"] = y_true.astype(int)
    out["prob_malignant"] = y_prob.astype(float)
    out["pred_label"] = (y_prob >= threshold).astype(int)
    out["threshold"] = float(threshold)
    return out


def load_models(
    checkpoint_paths: list[Path],
    model_name: str,
    device: torch.device,
) -> list[nn.Module]:
    models_list: list[nn.Module] = []
    for ckpt_path in checkpoint_paths:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        model = build_model(model_name=model_name, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models_list.append(model)
    return models_list


def compute_ensemble_weights(
    checkpoint_paths: list[Path],
    weighting: str,
    ranking_df: pd.DataFrame | None = None,
) -> np.ndarray:
    if weighting == "uniform":
        weights = np.ones(len(checkpoint_paths), dtype=np.float64)
        return weights / max(weights.sum(), 1.0)

    ranking_map: dict[str, float] = {}
    if ranking_df is not None and "val_auc" in ranking_df.columns:
        for _, row in ranking_df.iterrows():
            if "checkpoint_path" in ranking_df.columns:
                key = str(Path(str(row["checkpoint_path"])).expanduser().resolve())
                ranking_map[key] = float(row["val_auc"])
            if "checkpoint_name" in ranking_df.columns:
                ranking_map[str(row["checkpoint_name"])] = float(row["val_auc"])

    raw: list[float] = []
    for ckpt_path in checkpoint_paths:
        key_abs = str(ckpt_path.resolve())
        key_name = ckpt_path.name

        if key_abs in ranking_map:
            score = ranking_map[key_abs]
        elif key_name in ranking_map:
            score = ranking_map[key_name]
        else:
            ckpt = load_checkpoint(ckpt_path, map_location="cpu")
            score = float(ckpt.get("metrics", {}).get("val_auc", 0.0))
        if not np.isfinite(score):
            score = 0.0
        raw.append(max(score, 0.0))

    weights = np.asarray(raw, dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(checkpoint_paths), dtype=np.float64)
    weights /= float(weights.sum())
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier ensemble with optional TTA and threshold sweep.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", default="", help="Run dir containing checkpoints (used with --checkpoint-names).")
    parser.add_argument(
        "--checkpoint-names",
        default="best_auc.pt,last.pt",
        help="Comma-separated checkpoint filenames in <run-dir>/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        default="",
        help="Optional comma-separated absolute checkpoint paths. Overrides run-dir/checkpoint-names.",
    )
    parser.add_argument(
        "--ranking-csv",
        default="",
        help="Optional ranking CSV from select_classifier_checkpoint.py. If set, checkpoints are taken from it.",
    )
    parser.add_argument("--top-k-from-ranking", type=int, default=0, help="Use top-K rows from ranking CSV (0 = all).")
    parser.add_argument(
        "--model",
        required=True,
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "resnet18"],
    )
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument(
        "--ensemble-weighting",
        default="uniform",
        choices=["uniform", "val_auc"],
        help="How to weight checkpoints in ensemble.",
    )
    parser.add_argument("--threshold-mode", default="recall_floor", choices=["recall_floor", "max_f1"])
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--cache-dir", default="", help="Optional cache directory for eval tensors")
    parser.add_argument("--out-dir", default="", help="Default: <run-dir>/analysis/ensemble_eval")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    required_cols = {"dcm_path", "label", "split"}
    if not required_cols.issubset(manifest.columns):
        missing = sorted(required_cols - set(manifest.columns))
        raise ValueError(f"Manifest missing columns: {missing}")

    target_recall = float(min(max(args.target_recall, 0.0), 1.0))
    threshold_min = float(min(max(args.threshold_min, 0.0), 1.0))
    threshold_max = float(min(max(args.threshold_max, 0.0), 1.0))
    threshold_steps = max(int(args.threshold_steps), 2)

    ranking_df: pd.DataFrame | None = None
    if args.ranking_csv:
        ranking_path = Path(args.ranking_csv).expanduser().resolve()
        ranking_df = pd.read_csv(ranking_path)
        if ranking_df.empty:
            raise RuntimeError(f"Ranking CSV is empty: {ranking_path}")
        if args.top_k_from_ranking > 0:
            ranking_df = ranking_df.head(args.top_k_from_ranking).copy()

        if "checkpoint_path" in ranking_df.columns:
            checkpoint_paths = [Path(str(x)).expanduser().resolve() for x in ranking_df["checkpoint_path"].tolist()]
        elif "checkpoint_name" in ranking_df.columns:
            if not args.run_dir:
                raise ValueError("When ranking CSV has checkpoint_name only, provide --run-dir.")
            ckpt_dir = Path(args.run_dir).expanduser().resolve() / "checkpoints"
            checkpoint_paths = [ckpt_dir / str(x) for x in ranking_df["checkpoint_name"].tolist()]
        else:
            raise ValueError("Ranking CSV must contain `checkpoint_path` or `checkpoint_name` column.")

        if args.run_dir:
            base_run_dir = Path(args.run_dir).expanduser().resolve()
        else:
            base_run_dir = checkpoint_paths[0].parent.parent
    elif args.checkpoint_paths:
        checkpoint_paths = [Path(x.strip()).expanduser().resolve() for x in args.checkpoint_paths.split(",") if x.strip()]
        base_run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else checkpoint_paths[0].parent.parent
    else:
        if not args.run_dir:
            raise ValueError("Provide --run-dir when using --checkpoint-names.")
        base_run_dir = Path(args.run_dir).expanduser().resolve()
        ckpt_dir = base_run_dir / "checkpoints"
        names = [x.strip() for x in args.checkpoint_names.split(",") if x.strip()]
        checkpoint_paths = [(ckpt_dir / n).resolve() for n in names]

    if not checkpoint_paths:
        raise RuntimeError("No checkpoints specified.")
    checkpoint_paths = list(dict.fromkeys([p.resolve() for p in checkpoint_paths]))
    for p in checkpoint_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else base_run_dir / "analysis" / "ensemble_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    val_df = manifest[manifest["split"] == "val"].copy().reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    if val_df.empty or test_df.empty:
        raise RuntimeError("Manifest must contain non-empty val and test splits.")

    val_ds = EvalDicomDataset(val_df, image_size=args.image_size, cache_dir=(cache_dir / "val") if cache_dir else None)
    test_ds = EvalDicomDataset(test_df, image_size=args.image_size, cache_dir=(cache_dir / "test") if cache_dir else None)

    device = pick_device(args.device)
    use_pin_memory = device.type == "cuda"
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

    models_list = load_models(checkpoint_paths=checkpoint_paths, model_name=args.model, device=device)
    weights = compute_ensemble_weights(
        checkpoint_paths=checkpoint_paths,
        weighting=args.ensemble_weighting,
        ranking_df=ranking_df,
    )

    val_idx, val_y_true, val_y_prob = eval_split_ensemble(
        loader=val_loader,
        models=models_list,
        weights=weights,
        device=device,
        tta_hflip=args.tta_hflip,
    )
    test_idx, test_y_true, test_y_prob = eval_split_ensemble(
        loader=test_loader,
        models=models_list,
        weights=weights,
        device=device,
        tta_hflip=args.tta_hflip,
    )

    sweep_df = threshold_sweep(
        y_true=val_y_true,
        y_prob=val_y_prob,
        t_min=threshold_min,
        t_max=threshold_max,
        t_steps=threshold_steps,
    )
    selected_threshold, selection_reason = pick_threshold(sweep_df=sweep_df, mode=args.threshold_mode, target_recall=target_recall)

    val_metrics = compute_binary_metrics(y_true=val_y_true, y_prob=val_y_prob, threshold=selected_threshold)
    test_metrics = compute_binary_metrics(y_true=test_y_true, y_prob=test_y_prob, threshold=selected_threshold)

    val_pred_df = attach_predictions(val_df, val_idx, val_y_true, val_y_prob, selected_threshold)
    test_pred_df = attach_predictions(test_df, test_idx, test_y_true, test_y_prob, selected_threshold)

    sweep_df.to_csv(out_dir / "val_threshold_sweep.csv", index=False)
    val_pred_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "run_dir": str(base_run_dir),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "model": args.model,
        "image_size": int(args.image_size),
        "device": str(device),
        "ranking_csv": str(Path(args.ranking_csv).expanduser().resolve()) if args.ranking_csv else None,
        "top_k_from_ranking": int(args.top_k_from_ranking),
        "tta_hflip": bool(args.tta_hflip),
        "ensemble_weighting": str(args.ensemble_weighting),
        "ensemble_weights": [float(x) for x in weights.tolist()],
        "threshold_mode": args.threshold_mode,
        "target_recall": target_recall,
        "selected_threshold": float(selected_threshold),
        "selection_reason": selection_reason,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "artifacts": {
            "val_threshold_sweep": str((out_dir / "val_threshold_sweep.csv").resolve()),
            "val_predictions": str((out_dir / "val_predictions.csv").resolve()),
            "test_predictions": str((out_dir / "test_predictions.csv").resolve()),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
