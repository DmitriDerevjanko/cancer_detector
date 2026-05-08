#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointBundle:
    epoch: int
    global_step: int
    best_metric: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None


def ensure_run_dirs(run_dir: Path) -> tuple[Path, Path, Path]:
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, logs_dir


def _atomic_write_json(payload: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target_path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        temp_name = tmp.name
    os.replace(temp_name, target_path)


def save_run_state(run_dir: Path, payload: dict[str, Any]) -> None:
    state_path = run_dir / "state.json"
    _atomic_write_json(payload=payload, target_path=state_path)


def load_run_state(run_dir: Path) -> dict[str, Any] | None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_torch_atomic(checkpoint: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target_path.parent)) as tmp:
        torch.save(checkpoint, tmp.name)
        temp_name = tmp.name
    os.replace(temp_name, target_path)


def save_checkpoint(
    *,
    ckpt_dir: Path,
    bundle: CheckpointBundle,
    metrics: dict[str, float],
    file_name: str,
) -> Path:
    target_path = ckpt_dir / file_name
    checkpoint = {
        "epoch": bundle.epoch,
        "global_step": bundle.global_step,
        "best_metric": bundle.best_metric,
        "metrics": metrics,
        "model_state_dict": bundle.model_state_dict,
        "optimizer_state_dict": bundle.optimizer_state_dict,
        "scheduler_state_dict": bundle.scheduler_state_dict,
        "scaler_state_dict": bundle.scaler_state_dict,
    }
    _save_torch_atomic(checkpoint=checkpoint, target_path=target_path)
    return target_path


def load_checkpoint(checkpoint_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def append_metrics_row(logs_dir: Path, row: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "metrics.csv"

    headers = list(row.keys())
    values = [str(row[h]) for h in headers]

    if not csv_path.exists():
        csv_path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")
        return

    with csv_path.open("a", encoding="utf-8") as f:
        f.write(",".join(values) + "\n")
