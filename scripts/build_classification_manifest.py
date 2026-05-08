#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

UID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def normalize_label(value: object) -> str | None:
    text = str(value).strip().lower()
    if "malig" in text:
        return "malignant"
    if "benig" in text:
        return "benign"
    return None


def extract_uid_pair(path_text: str) -> tuple[str, str] | None:
    tokens = [t.strip() for t in str(path_text).replace("\\", "/").split("/") if t.strip()]
    uids = [t for t in tokens if UID_PATTERN.fullmatch(t)]
    if len(uids) < 2:
        return None
    return uids[0], uids[1]


def infer_source_split(csv_name: str) -> str:
    name = csv_name.lower()
    if "test" in name:
        return "test"
    return "train"


def safe_str(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def split_train_val_by_patient(df: pd.DataFrame, val_size: float, seed: int) -> dict[str, str]:
    patient_flags = (
        df.groupby("patient_id")["label"]
        .apply(lambda s: "malignant" if (s == "malignant").any() else "benign")
        .reset_index(name="label")
    )

    train_patients, val_patients = train_test_split(
        patient_flags,
        test_size=val_size,
        random_state=seed,
        stratify=patient_flags["label"],
    )

    mapping = {pid: "train" for pid in train_patients["patient_id"].tolist()}
    mapping.update({pid: "val" for pid in val_patients["patient_id"].tolist()})
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Build classification manifest from CBIS-DDSM CSV + DICOM index")
    parser.add_argument("--source-root", default="data/raw/cbis_ddsm_tcia", help="Root directory with csv/ and index/")
    parser.add_argument("--index-csv", default="", help="Override dicom index csv path")
    parser.add_argument(
        "--path-column",
        default="image file path",
        help="CSV column to map to DICOM files (recommended: image file path)",
    )
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation fraction from train rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="data/processed/classification_manifest_full.csv",
        help="Output manifest CSV",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    csv_root = source_root / "csv"
    index_csv = Path(args.index_csv).expanduser().resolve() if args.index_csv else source_root / "index" / "dicom_index.csv"
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_root.exists():
        raise FileNotFoundError(f"CSV root not found: {csv_root}")
    if not index_csv.exists():
        raise FileNotFoundError(f"DICOM index not found: {index_csv}")

    index_df = pd.read_csv(index_csv, dtype=str)
    required_index_cols = {"StudyInstanceUID", "SeriesInstanceUID", "file_path"}
    if not required_index_cols.issubset(index_df.columns):
        missing = sorted(required_index_cols - set(index_df.columns))
        raise ValueError(f"Missing index columns: {missing}")

    pair_to_path: dict[tuple[str, str], str] = {}
    dup_pairs = 0

    for _, row in index_df.iterrows():
        key = (safe_str(row["StudyInstanceUID"]), safe_str(row["SeriesInstanceUID"]))
        path = safe_str(row["file_path"])
        if not key[0] or not key[1] or not path:
            continue
        if key in pair_to_path:
            dup_pairs += 1
            continue
        pair_to_path[key] = path

    records: list[dict[str, str | int]] = []
    missing_uid = 0
    missing_pair = 0

    csv_files = sorted(csv_root.glob("*case_description*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No case description CSV files in {csv_root}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if args.path_column not in df.columns:
            raise ValueError(f"Column '{args.path_column}' missing in {csv_path.name}")
        if "pathology" not in df.columns:
            raise ValueError(f"Column 'pathology' missing in {csv_path.name}")

        source_split = infer_source_split(csv_path.name)

        for _, row in df.iterrows():
            label = normalize_label(row.get("pathology"))
            if label is None:
                continue

            raw_path = safe_str(row.get(args.path_column, ""))
            uid_pair = extract_uid_pair(raw_path)
            if uid_pair is None:
                missing_uid += 1
                continue

            dcm_path = pair_to_path.get(uid_pair)
            if dcm_path is None:
                missing_pair += 1
                continue

            patient_id = safe_str(row.get("patient_id", ""))
            side = safe_str(row.get("left or right breast", ""))
            view = safe_str(row.get("image view", ""))
            abn_id = safe_str(row.get("abnormality id", ""))
            case_id = "_".join([x for x in [patient_id, side, view, abn_id] if x])
            if not case_id:
                case_id = patient_id or "unknown_case"

            records.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "label": label,
                    "pathology_raw": safe_str(row.get("pathology", "")),
                    "source_csv": csv_path.name,
                    "source_split": source_split,
                    "path_column": args.path_column,
                    "raw_path": raw_path,
                    "study_uid": uid_pair[0],
                    "series_uid": uid_pair[1],
                    "dcm_path": dcm_path,
                }
            )

    if not records:
        raise RuntimeError("No manifest records created.")

    manifest = pd.DataFrame.from_records(records)
    manifest = manifest.drop_duplicates(subset=["dcm_path"]).reset_index(drop=True)

    train_rows = manifest[manifest["source_split"] == "train"].copy()
    test_rows = manifest[manifest["source_split"] == "test"].copy()

    if train_rows.empty or test_rows.empty:
        raise RuntimeError("Expected both train and test rows from source CSV files.")

    split_map = split_train_val_by_patient(df=train_rows, val_size=args.val_size, seed=args.seed)
    train_rows["split"] = train_rows["patient_id"].map(split_map)
    train_rows["split"] = train_rows["split"].fillna("train")
    test_rows["split"] = "test"

    out_df = pd.concat([train_rows, test_rows], ignore_index=True)
    out_df.to_csv(out_path, index=False)

    split_summary = out_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("Split summary:")
    print(split_summary)
    print(f"Rows: {len(out_df)}")
    print(f"Unique patients: {out_df['patient_id'].nunique()}")
    print(f"Missing UID rows: {missing_uid}")
    print(f"Missing index-pair rows: {missing_pair}")
    print(f"Duplicate UID pairs skipped in index map: {dup_pairs}")
    print(f"Saved manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
