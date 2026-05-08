#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def to_display(arr: np.ndarray, photometric: str | None) -> np.ndarray:
    image = arr.astype(np.float32)
    if photometric == "MONOCHROME1":
        image = image.max() - image
    min_v = float(image.min())
    max_v = float(image.max())
    if max_v > min_v:
        image = (image - min_v) / (max_v - min_v)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview DICOM files as images")
    parser.add_argument(
        "--root",
        default="data/raw/cbis_ddsm_tcia/dicom",
        help="Directory with .dcm files",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Filter by SeriesDescription substring (case-insensitive), e.g. 'full mammogram'",
    )
    parser.add_argument("--n", type=int, default=6, help="Number of samples to display")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="", help="Optional path to save preview PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    dcm_files = sorted(root.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {root}")

    filtered: list[Path] = []
    needle = args.contains.strip().lower()

    for path in dcm_files:
        if not needle:
            filtered.append(path)
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        desc = str(getattr(ds, "SeriesDescription", "")).lower()
        if needle in desc:
            filtered.append(path)

    if not filtered:
        raise RuntimeError(f"No files matched --contains='{args.contains}'")

    rng = random.Random(args.seed)
    sample = rng.sample(filtered, k=min(args.n, len(filtered)))

    cols = 3
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes_arr = np.array(axes).reshape(-1)

    for ax in axes_arr:
        ax.axis("off")

    for ax, path in zip(axes_arr, sample):
        ds = pydicom.dcmread(path)
        image = to_display(ds.pixel_array, getattr(ds, "PhotometricInterpretation", None))
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{path.name}\n{getattr(ds, 'PatientID', 'NA')} | {getattr(ds, 'SeriesDescription', 'NA')}",
            fontsize=8,
        )
        ax.axis("off")

    fig.tight_layout()

    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Saved preview: {out}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
