#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_files(root: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        count = 0
        for p in label_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                count += 1
        stats[label_dir.name] = count
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple dataset report")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    stats = {
        "root": str(root),
        "labels": count_files(root),
    }
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
