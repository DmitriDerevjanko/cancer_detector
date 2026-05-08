#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits by case_id")
    parser.add_argument("--manifest", required=True, help="Path to subset_manifest.csv")
    parser.add_argument("--out", required=True, help="Output CSV path with split column")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"case_id", "label"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    case_df = df[["case_id", "label"]].drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    train_val, test = train_test_split(
        case_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=case_df["label"],
    )

    val_ratio = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=train_val["label"],
    )

    split_map = {**{c: "train" for c in train["case_id"]}, **{c: "val" for c in val["case_id"]}, **{c: "test" for c in test["case_id"]}}

    out_df = df.copy()
    out_df["split"] = out_df["case_id"].map(split_map)
    out_df.to_csv(out_path, index=False)

    summary = out_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("Split summary (rows):")
    print(summary)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
