#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build API inference config files from evaluation summaries.")
    parser.add_argument("--balanced-summary", required=True, help="checkpoint_selection summary.json path")
    parser.add_argument("--ensemble-summary", required=True, help="ensemble_eval summary.json path")
    parser.add_argument("--model", required=True, choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "resnet18"])
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--out-dir", required=True, help="Output dir for inference_config*.json")
    parser.add_argument("--balanced-threshold", type=float, default=-1.0, help="Override balanced threshold")
    parser.add_argument("--high-recall-threshold", type=float, default=-1.0, help="Override high-recall threshold")
    parser.add_argument("--balanced-note", default="Balanced operating point selected from checkpoint ranking.")
    parser.add_argument("--high-recall-note", default="High-recall operating point with ensemble and optional TTA.")
    args = parser.parse_args()

    balanced_summary_path = Path(args.balanced_summary).expanduser().resolve()
    ensemble_summary_path = Path(args.ensemble_summary).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not balanced_summary_path.exists():
        raise FileNotFoundError(f"Balanced summary not found: {balanced_summary_path}")
    if not ensemble_summary_path.exists():
        raise FileNotFoundError(f"Ensemble summary not found: {ensemble_summary_path}")

    bs = _load_json(balanced_summary_path)
    es = _load_json(ensemble_summary_path)

    champion = bs.get("champion", {})
    checkpoint = Path(str(champion.get("checkpoint_path", ""))).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Champion checkpoint missing: {checkpoint}")

    balanced_threshold = float(champion.get("selected_threshold", 0.5))
    if args.balanced_threshold >= 0.0:
        balanced_threshold = float(args.balanced_threshold)

    checkpoints = [Path(str(x)).expanduser().resolve() for x in es.get("checkpoints", [])]
    if not checkpoints:
        raise RuntimeError("Ensemble summary has no checkpoints.")
    for p in checkpoints:
        if not p.exists():
            raise FileNotFoundError(f"Ensemble checkpoint missing: {p}")

    ensemble_weights = es.get("ensemble_weights", [])
    if not isinstance(ensemble_weights, list) or len(ensemble_weights) != len(checkpoints):
        ensemble_weights = [1.0 / float(len(checkpoints))] * len(checkpoints)
    else:
        s = sum(float(x) for x in ensemble_weights)
        if s <= 0.0:
            ensemble_weights = [1.0 / float(len(checkpoints))] * len(checkpoints)
        else:
            ensemble_weights = [float(x) / float(s) for x in ensemble_weights]

    high_recall_threshold = float(es.get("selected_threshold", 0.5))
    if args.high_recall_threshold >= 0.0:
        high_recall_threshold = float(args.high_recall_threshold)

    balanced_cfg = {
        "checkpoint": str(checkpoint),
        "model": args.model,
        "image_size": int(args.image_size),
        "threshold": float(balanced_threshold),
        "selection_summary": str(balanced_summary_path),
        "note": str(args.balanced_note),
    }
    high_recall_cfg = {
        "checkpoint": str(checkpoints[0]),
        "ensemble_checkpoints": [str(p) for p in checkpoints],
        "ensemble_weights": [float(x) for x in ensemble_weights],
        "model": args.model,
        "image_size": int(args.image_size),
        "tta_hflip": bool(es.get("tta_hflip", False)),
        "threshold": float(high_recall_threshold),
        "target_recall": float(es.get("target_recall", 0.80)),
        "selection_summary": str(ensemble_summary_path),
        "note": str(args.high_recall_note),
    }

    balanced_path = out_dir / "inference_config.json"
    high_recall_path = out_dir / "inference_config_high_recall.json"
    balanced_path.write_text(json.dumps(balanced_cfg, indent=2), encoding="utf-8")
    high_recall_path.write_text(json.dumps(high_recall_cfg, indent=2), encoding="utf-8")

    print(json.dumps({"balanced": str(balanced_path), "high_recall": str(high_recall_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

