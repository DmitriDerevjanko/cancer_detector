#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_CANDIDATES = ["pathology", "label", "diagnosis", "target", "class"]
CASE_ID_CANDIDATES = ["patient_id", "subject_id", "study_id", "image file path", "patient_id"]
UID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def normalize_label(value: object) -> str | None:
    text = str(value).strip().lower()
    if "malig" in text:
        return "malignant"
    if "benig" in text:
        return "benign"
    return None


def sanitize_case_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned[:120] if cleaned else "unknown_case"


def discover_csvs(source_root: Path) -> list[Path]:
    csv_files = sorted(source_root.rglob("*.csv"))
    preferred = [p for p in csv_files if "case_description" in p.name.lower()]
    return preferred or csv_files


def pick_label_column(df: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in LABEL_CANDIDATES:
        if c in lower:
            return lower[c]
    return None


def pick_case_id(row: pd.Series, index: int) -> str:
    row_keys = {k.lower(): k for k in row.index}
    for candidate in CASE_ID_CANDIDATES:
        if candidate in row_keys:
            value = str(row[row_keys[candidate]])
            if value and value.lower() != "nan":
                return sanitize_case_id(value)
    return f"case_{index:06d}"


def resolve_candidate_path(raw: str, source_root: Path) -> Path | None:
    candidate = Path(raw.strip().replace("\\", os.sep))
    tries = []
    if candidate.is_absolute():
        tries.append(candidate)
    tries.append(source_root / candidate)
    tries.append(source_root / candidate.name)

    for p in tries:
        if p.exists():
            return p
    return None


def extract_uid_pair(raw: str) -> tuple[str, str] | None:
    tokens = [t.strip() for t in raw.replace("\\", "/").split("/") if t.strip()]
    uids = [t for t in tokens if UID_PATTERN.fullmatch(t)]
    if len(uids) < 2:
        return None
    return uids[0], uids[1]


def build_uid_to_files_map(source_root: Path) -> dict[tuple[str, str], list[Path]]:
    index_csv = source_root / "index" / "dicom_index.csv"
    if not index_csv.exists():
        return {}

    df = pd.read_csv(index_csv, dtype=str)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "file_path"}
    if not required.issubset(set(df.columns)):
        return {}

    mapping: dict[tuple[str, str], list[Path]] = {}
    for _, row in df.iterrows():
        study_uid = str(row["StudyInstanceUID"]).strip()
        series_uid = str(row["SeriesInstanceUID"]).strip()
        file_path = Path(str(row["file_path"]).strip())
        if not study_uid or not series_uid:
            continue
        if not file_path.exists():
            continue
        key = (study_uid, series_uid)
        mapping.setdefault(key, []).append(file_path)

    for key in mapping:
        mapping[key] = sorted(set(mapping[key]))
    return mapping


def collect_row_files(
    row: pd.Series,
    source_root: Path,
    uid_to_files_map: dict[tuple[str, str], list[Path]],
) -> list[Path]:
    files: list[Path] = []
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        raw = value.strip()
        if len(raw) < 3:
            continue

        key_l = str(key).lower()
        looks_like_path = any(raw.lower().endswith(ext) for ext in IMAGE_EXTS) or "/" in raw or "\\" in raw
        looks_relevant_col = any(k in key_l for k in ["image", "mask", "roi", "crop", "file", "path"])
        if not (looks_like_path and looks_relevant_col):
            continue

        path = resolve_candidate_path(raw, source_root)
        if path is not None:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                files.append(path)
            elif path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        files.append(f)
            continue

        uid_pair = extract_uid_pair(raw)
        if uid_pair and uid_pair in uid_to_files_map:
            files.extend(uid_to_files_map[uid_pair])

    unique = sorted(set(files))
    return unique


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a mini stratified CBIS-DDSM subset.")
    parser.add_argument("--source-root", required=True, help="Root directory where raw dataset is stored")
    parser.add_argument("--out-root", required=True, help="Output directory for subset")
    parser.add_argument("--per-class", type=int, default=80, help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    uid_to_files_map = build_uid_to_files_map(source_root=source_root)

    csv_files = discover_csvs(source_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_root}")

    tables = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        label_col = pick_label_column(df)
        if not label_col:
            continue

        df = df.copy()
        df["__label"] = df[label_col].map(normalize_label)
        df = df[df["__label"].notna()]
        if df.empty:
            continue
        df["__source_csv"] = str(csv_path)
        tables.append(df)

    if not tables:
        raise RuntimeError("No labeled rows found in CSV files.")

    all_rows = pd.concat(tables, ignore_index=True)

    sampled_parts = []
    for label in ["benign", "malignant"]:
        group = all_rows[all_rows["__label"] == label]
        if group.empty:
            continue
        sampled_parts.append(group.sample(n=min(args.per_class, len(group)), random_state=args.seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    records = []
    skipped = 0

    for idx, row in sampled.iterrows():
        label = row["__label"]
        case_id = pick_case_id(row, idx)
        files = collect_row_files(row, source_root, uid_to_files_map)
        if not files:
            skipped += 1
            continue

        case_dir = out_root / label / case_id
        count = 0
        for src in files:
            try:
                rel = src.relative_to(source_root)
                dst = case_dir / rel
            except ValueError:
                dst = case_dir / src.name

            link_or_copy(src=src, dst=dst, mode=args.link_mode)
            count += 1

        records.append(
            {
                "case_id": case_id,
                "label": label,
                "n_files": count,
                "source_csv": row["__source_csv"],
                "output_dir": str(case_dir),
            }
        )

    manifest_df = pd.DataFrame.from_records(records)
    manifest_path = out_root / "subset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "samples_requested_per_class": args.per_class,
        "rows_sampled": int(len(sampled)),
        "cases_written": int(len(records)),
        "rows_skipped_no_files": int(skipped),
        "link_mode": args.link_mode,
        "uid_index_pairs": int(len(uid_to_files_map)),
        "labels_written": manifest_df["label"].value_counts().to_dict() if not manifest_df.empty else {},
    }
    (out_root / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
