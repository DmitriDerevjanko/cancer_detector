#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_preferred_file(files: list[Path], desc_map: dict[str, str]) -> tuple[Path, str]:
    ranked: list[tuple[int, Path, str]] = []
    for f in files:
        desc = desc_map.get(f.name, "")
        dl = desc.lower()
        if "full mammogram" in dl:
            rank = 0
        elif "cropped" in dl:
            rank = 1
        elif "roi" in dl or "mask" in dl:
            rank = 3
        else:
            rank = 2
        ranked.append((rank, f, desc))

    ranked.sort(key=lambda x: (x[0], str(x[1])))
    _, file_path, desc = ranked[0]
    return file_path, desc


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-ready mini classification manifest")
    parser.add_argument(
        "--subset-manifest",
        default="data/interim/cbis_ddsm_mini/subset_manifest_splits.csv",
        help="Mini subset manifest with split column",
    )
    parser.add_argument(
        "--index-csv",
        default="data/raw/cbis_ddsm_tcia/index/dicom_index.csv",
        help="DICOM index CSV",
    )
    parser.add_argument(
        "--out",
        default="data/processed/classification_manifest_mini.csv",
        help="Output manifest path",
    )
    args = parser.parse_args()

    subset_manifest = Path(args.subset_manifest).expanduser().resolve()
    index_csv = Path(args.index_csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not subset_manifest.exists():
        raise FileNotFoundError(subset_manifest)
    if not index_csv.exists():
        raise FileNotFoundError(index_csv)

    subset_df = pd.read_csv(subset_manifest)
    required_subset = {"case_id", "label", "output_dir", "split"}
    if not required_subset.issubset(subset_df.columns):
        missing = sorted(required_subset - set(subset_df.columns))
        raise ValueError(f"Subset manifest missing columns: {missing}")

    idx_df = pd.read_csv(index_csv, dtype=str)
    if "file_name" not in idx_df.columns or "SeriesDescription" not in idx_df.columns:
        raise ValueError("Index CSV must include file_name and SeriesDescription")

    desc_map = {
        str(row["file_name"]): str(row.get("SeriesDescription", "") or "")
        for _, row in idx_df.iterrows()
    }

    records: list[dict[str, str]] = []
    missing_files = 0

    for _, row in subset_df.iterrows():
        case_id = str(row["case_id"])
        label = str(row["label"])
        split = str(row["split"])
        output_dir = Path(str(row["output_dir"])).expanduser().resolve()

        files = sorted([p for p in output_dir.rglob("*.dcm") if p.is_file()])
        if not files:
            missing_files += 1
            continue

        best_file, best_desc = pick_preferred_file(files=files, desc_map=desc_map)

        records.append(
            {
                "case_id": case_id,
                "patient_id": case_id,
                "label": label,
                "split": split,
                "dcm_path": str(best_file),
                "series_description": best_desc,
            }
        )

    if not records:
        raise RuntimeError("No records created for mini manifest")

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_path, index=False)

    print("Split summary:")
    print(out_df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"Rows: {len(out_df)}")
    print(f"Missing rows without DICOM: {missing_files}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
