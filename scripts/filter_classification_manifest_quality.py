#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def _load_normalized_dicom(dcm_path: Path) -> np.ndarray:
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

    return arr


def _compute_quality_features(arr: np.ndarray) -> dict[str, float]:
    p1, p99 = np.percentile(arr, [1, 99])
    return {
        "quality_tissue_ratio": float((arr > 0.08).mean()),
        "quality_std": float(arr.std()),
        "quality_dyn_p99_p1": float(p99 - p1),
        "quality_white_ratio": float((arr > 0.95).mean()),
    }


def _print_summary(title: str, frame: pd.DataFrame, split_col: str, label_col: str) -> None:
    print(f"\n{title}")
    print(f"Rows: {len(frame)}")
    if split_col in frame.columns and label_col in frame.columns:
        print(frame.groupby([split_col, label_col]).size().rename("count").to_string())
    elif split_col in frame.columns:
        print(frame.groupby(split_col).size().rename("count").to_string())
    elif label_col in frame.columns:
        print(frame.groupby(label_col).size().rename("count").to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter classification manifest by simple DICOM quality metrics")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV (must include dcm_path column)")
    parser.add_argument("--out", required=True, help="Output filtered manifest CSV")
    parser.add_argument("--stats-out", default="", help="Optional full manifest with computed quality columns")
    parser.add_argument("--dropped-out", default="", help="Optional dropped rows with quality columns")
    parser.add_argument("--dcm-column", default="dcm_path")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--min-tissue-ratio", type=float, default=0.001)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--min-dyn-p99-p1", type=float, default=0.0)
    parser.add_argument(
        "--train-quantile",
        type=float,
        default=0.05,
        help="If in [0,1), raise thresholds to this quantile computed on train split quality stats",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    stats_out_path = Path(args.stats_out).expanduser().resolve() if args.stats_out else None
    dropped_out_path = Path(args.dropped_out).expanduser().resolve() if args.dropped_out else None

    df = pd.read_csv(manifest_path)
    if args.dcm_column not in df.columns:
        raise ValueError(f"Manifest is missing required column: {args.dcm_column}")

    rows: list[dict[str, float]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing quality metrics"):
        dcm_path = Path(str(getattr(row, args.dcm_column))).expanduser().resolve()
        arr = _load_normalized_dicom(dcm_path=dcm_path)
        features = _compute_quality_features(arr=arr)
        rows.append(features)

    qdf = pd.DataFrame(rows)
    merged = pd.concat([df.reset_index(drop=True), qdf], axis=1)

    tissue_thr = float(max(args.min_tissue_ratio, 0.0))
    std_thr = float(max(args.min_std, 0.0))
    dyn_thr = float(max(args.min_dyn_p99_p1, 0.0))

    if 0.0 <= float(args.train_quantile) < 1.0:
        if args.split_column in merged.columns:
            train_frame = merged[merged[args.split_column].astype(str) == "train"]
        else:
            train_frame = merged
        if not train_frame.empty:
            q = float(args.train_quantile)
            tissue_thr = max(tissue_thr, float(train_frame["quality_tissue_ratio"].quantile(q)))
            std_thr = max(std_thr, float(train_frame["quality_std"].quantile(q)))
            dyn_thr = max(dyn_thr, float(train_frame["quality_dyn_p99_p1"].quantile(q)))

    keep_mask = (
        (merged["quality_tissue_ratio"] >= tissue_thr)
        & (merged["quality_std"] >= std_thr)
        & (merged["quality_dyn_p99_p1"] >= dyn_thr)
    )

    kept = merged[keep_mask].copy()
    dropped = merged[~keep_mask].copy()

    print(
        f"\nThresholds used: tissue_ratio>={tissue_thr:.6f}, std>={std_thr:.6f}, dyn_p99_p1>={dyn_thr:.6f}"
    )
    _print_summary("Original", merged, split_col=args.split_column, label_col=args.label_column)
    _print_summary("Kept", kept, split_col=args.split_column, label_col=args.label_column)
    _print_summary("Dropped", dropped, split_col=args.split_column, label_col=args.label_column)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept[df.columns].to_csv(out_path, index=False)
    print(f"\nSaved filtered manifest: {out_path}")

    if stats_out_path is not None:
        stats_out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(stats_out_path, index=False)
        print(f"Saved manifest + quality stats: {stats_out_path}")

    if dropped_out_path is not None:
        dropped_out_path.parent.mkdir(parents=True, exist_ok=True)
        dropped.to_csv(dropped_out_path, index=False)
        print(f"Saved dropped rows: {dropped_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
