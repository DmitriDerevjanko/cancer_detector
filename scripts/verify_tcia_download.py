#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_CSV_HINTS = [
    "mass_case_description_train",
    "calc_case_description_train",
    "mass_case_description_test",
    "calc_case_description_test",
]


def bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 2)


def find_dicom_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local TCIA CBIS-DDSM download")
    parser.add_argument("--root", default="data/raw/cbis_ddsm_tcia", help="TCIA dataset root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    csv_root = root / "csv"
    if csv_root.exists():
        csv_files = sorted(csv_root.rglob("*.csv"))
    else:
        csv_files = sorted(root.rglob("*.csv"))
    dcm_files = find_dicom_files(root)

    missing_csv_hints: list[str] = []
    csv_names_lower = [p.name.lower() for p in csv_files]
    for hint in REQUIRED_CSV_HINTS:
        if not any(hint in name for name in csv_names_lower):
            missing_csv_hints.append(hint)

    total_dcm_size = sum(p.stat().st_size for p in dcm_files)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dcm_count": len(dcm_files),
        "dcm_total_size_gb": bytes_to_gb(total_dcm_size),
        "csv_count": len(csv_files),
        "csv_files": [str(p) for p in csv_files],
        "required_csv_hints_missing": missing_csv_hints,
        "ready_for_next_step": len(dcm_files) > 0 and len(missing_csv_hints) == 0,
    }

    index_dir = root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    report_path = index_dir / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
