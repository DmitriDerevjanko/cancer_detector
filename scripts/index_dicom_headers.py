#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

FIELDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "Modality",
    "Laterality",
    "ImageLaterality",
    "ViewPosition",
    "Rows",
    "Columns",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Index DICOM headers into CSV")
    parser.add_argument("--root", default="data/raw/cbis_ddsm_tcia/dicom", help="Directory with .dcm")
    parser.add_argument(
        "--out",
        default="data/raw/cbis_ddsm_tcia/index/dicom_index.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(root.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {root}")

    records = []
    bad = 0

    for path in tqdm(files, desc="Indexing DICOM", unit="file"):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError):
            bad += 1
            continue

        row = {
            "file_name": path.name,
            "file_path": str(path),
        }
        for f in FIELDS:
            row[f] = str(getattr(ds, f, ""))
        records.append(row)

    df = pd.DataFrame.from_records(records)
    df.to_csv(out, index=False)

    print(f"Indexed files: {len(df)}")
    print(f"Skipped invalid: {bad}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
