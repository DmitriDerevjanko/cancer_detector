from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageEnhance


def _normalize_to_unit(arr: np.ndarray, mono1: bool = False) -> np.ndarray:
    x = arr.astype(np.float32)
    if mono1:
        x = x.max() - x
    min_v = float(x.min())
    max_v = float(x.max())
    if max_v > min_v:
        x = (x - min_v) / (max_v - min_v)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def _read_dicom_unit(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    mono1 = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    return _normalize_to_unit(arr, mono1=mono1)


def _read_image_unit(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _png_data_url_from_unit(unit: np.ndarray, max_size: int = 320) -> str:
    h, w = unit.shape[:2]
    scale = min(max_size / max(h, 1), max_size / max(w, 1), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    lo = float(np.percentile(unit, 1.0))
    hi = float(np.percentile(unit, 99.5))
    if hi > lo:
        x = (unit - lo) / (hi - lo)
    else:
        x = unit.copy()
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 0.9, dtype=np.float32)
    img8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img8, mode="L")
    pil = ImageEnhance.Sharpness(pil).enhance(1.15)
    pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    pil.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _image_quality_stats(unit: np.ndarray) -> dict[str, float]:
    fg_ratio = float((unit > 0.05).mean())
    p2 = float(np.percentile(unit, 2))
    p50 = float(np.percentile(unit, 50))
    p98 = float(np.percentile(unit, 98))
    contrast = max(p98 - p2, 0.0)

    fg_pref = max(0.0, 1.0 - abs(fg_ratio - 0.55) / 0.55)
    p50_pref = max(0.0, 1.0 - abs(p50 - 0.35) / 0.35)
    contrast_pref = max(0.0, 1.0 - abs(contrast - 0.60) / 0.60)
    quality_score = float(0.45 * fg_pref + 0.30 * p50_pref + 0.25 * contrast_pref)

    return {
        "fg_ratio": fg_ratio,
        "p50": p50,
        "contrast": contrast,
        "quality_score": quality_score,
    }


def _passes_visual_filter(stats: dict[str, float]) -> bool:
    # Remove very sparse/edge-only views and overfilled low-information frames.
    fg = float(stats["fg_ratio"])
    p50 = float(stats["p50"])
    contrast = float(stats["contrast"])
    return 0.08 <= fg <= 0.95 and p50 >= 0.05 and contrast >= 0.10


@dataclass(frozen=True)
class SampleItem:
    sample_id: str
    dcm_path: Path
    file_name: str
    ground_truth: str
    source_split: str
    thumbnail_png: str

    def to_public(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "file_name": self.file_name,
            "ground_truth": self.ground_truth,
            "source_split": self.source_split,
            "thumbnail_png": self.thumbnail_png,
        }


class SampleCatalog:
    def __init__(self) -> None:
        self._items: list[SampleItem] = []
        self._by_id: dict[str, SampleItem] = {}

    @staticmethod
    def _cache_path(manifest_path: Path) -> Path:
        return manifest_path.parent / f"{manifest_path.stem}_demo_samples_cache.json"

    @staticmethod
    def _load_cache(cache_path: Path, sample_count: int) -> list[dict[str, Any]] | None:
        if not cache_path.exists():
            return None
        try:
            payload = pd.read_json(cache_path)
        except Exception:
            return None
        if payload.empty:
            return None
        rows = payload.to_dict("records")
        if len(rows) < sample_count:
            return None
        valid: list[dict[str, Any]] = []
        for row in rows[:sample_count]:
            p = Path(str(row.get("dcm_path", ""))).expanduser().resolve()
            if not p.exists():
                return None
            valid.append(
                {
                    "label_norm": str(row.get("label_norm", "")).strip().lower(),
                    "split": str(row.get("split", "test")),
                    "dcm_path": p,
                }
            )
        return valid

    @staticmethod
    def _save_cache(cache_path: Path, selected: list[dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for cand in selected:
            row = cand["row"]
            rows.append(
                {
                    "label_norm": str(row.get("label_norm", "")),
                    "split": str(row.get("split", "test")),
                    "dcm_path": str(cand["dcm_path"]),
                }
            )
        try:
            pd.DataFrame(rows).to_json(cache_path, orient="records", indent=2)
        except Exception:
            return

    def load_from_manifest(self, manifest_path: Path, sample_count: int = 20, seed: int = 42) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Sample manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        required = {"dcm_path", "label", "split"}
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise ValueError(f"Sample manifest missing columns: {missing}")

        df = df.copy()
        df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
        df = df[df["label_norm"].isin({"benign", "malignant"})]
        if df.empty:
            raise RuntimeError("No eligible rows in sample manifest.")

        # Prefer test split for demo, fallback to all rows.
        test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
        base = test_df if not test_df.empty else df

        target = max(2, int(sample_count))
        half = target // 2
        cache_path = self._cache_path(manifest_path)

        cached_rows = self._load_cache(cache_path, sample_count=target)
        if cached_rows is not None:
            items: list[SampleItem] = []
            for idx, row in enumerate(cached_rows[:target], start=1):
                dcm_path = Path(row["dcm_path"]).expanduser().resolve()
                unit = _read_dicom_unit(dcm_path)
                thumb = _png_data_url_from_unit(unit, max_size=300)
                item = SampleItem(
                    sample_id=f"s{idx:03d}",
                    dcm_path=dcm_path,
                    file_name=dcm_path.name,
                    ground_truth=str(row["label_norm"]),
                    source_split=str(row.get("split", "test")),
                    thumbnail_png=thumb,
                )
                items.append(item)
            self._items = items
            self._by_id = {x.sample_id: x for x in items}
            return

        ranked_by_label: dict[str, list[dict[str, Any]]] = {"benign": [], "malignant": []}
        fallback_by_label: dict[str, list[dict[str, Any]]] = {"benign": [], "malignant": []}

        for row in base.to_dict("records"):
            label = str(row["label_norm"])
            dcm_path = Path(str(row["dcm_path"])).expanduser().resolve()
            if not dcm_path.exists():
                continue
            try:
                unit = _read_dicom_unit(dcm_path)
                stats = _image_quality_stats(unit)
                thumb = _png_data_url_from_unit(unit, max_size=300)
            except Exception:
                continue
            candidate = {
                "row": row,
                "dcm_path": dcm_path,
                "thumbnail_png": thumb,
                "stats": stats,
                "patient_id": str(row.get("patient_id", "")),
            }
            fallback_by_label[label].append(candidate)
            if _passes_visual_filter(stats):
                ranked_by_label[label].append(candidate)

        for label in ("benign", "malignant"):
            ranked_by_label[label].sort(key=lambda c: float(c["stats"]["quality_score"]), reverse=True)
            fallback_by_label[label].sort(key=lambda c: float(c["stats"]["quality_score"]), reverse=True)

        selected: list[dict[str, Any]] = []
        for label in ("benign", "malignant"):
            selected_label = list(ranked_by_label[label][:half])
            if len(selected_label) < half:
                need = half - len(selected_label)
                for cand in fallback_by_label[label][: half + need]:
                    if cand not in selected_label:
                        selected_label.append(cand)
                    if len(selected_label) >= half:
                        break
            selected.extend(selected_label[:half])

        # If odd target or any shortfall, top up from all labels by score.
        if len(selected) < target:
            pool = fallback_by_label["benign"] + fallback_by_label["malignant"]
            pool.sort(key=lambda c: float(c["stats"]["quality_score"]), reverse=True)
            for cand in pool:
                if cand in selected:
                    continue
                selected.append(cand)
                if len(selected) >= target:
                    break

        if selected:
            self._save_cache(cache_path, selected[:target])

        items: list[SampleItem] = []
        for idx, cand in enumerate(selected[:target], start=1):
            row = cand["row"]
            label = str(row["label_norm"])
            item = SampleItem(
                sample_id=f"s{idx:03d}",
                dcm_path=Path(cand["dcm_path"]),
                file_name=Path(cand["dcm_path"]).name,
                ground_truth=label,
                source_split=str(row.get("split", "unknown")),
                thumbnail_png=str(cand["thumbnail_png"]),
            )
            items.append(item)

        self._items = items
        self._by_id = {x.sample_id: x for x in items}

    def load_from_directory(self, sample_dir: Path, sample_count: int = 20) -> None:
        target = max(1, int(sample_count))
        paths = [sample_dir / f"s{idx:03d}.png" for idx in range(1, target + 1)]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Demo sample images missing from {sample_dir}: {missing}")

        items: list[SampleItem] = []
        for idx, path in enumerate(paths, start=1):
            unit = _read_image_unit(path)
            label = "benign" if idx <= 10 else "malignant"
            items.append(
                SampleItem(
                    sample_id=f"s{idx:03d}",
                    dcm_path=path.resolve(),
                    file_name=path.name,
                    ground_truth=label,
                    source_split="curated",
                    thumbnail_png=_png_data_url_from_unit(unit, max_size=300),
                )
            )

        self._items = items
        self._by_id = {x.sample_id: x for x in items}

    def list_public(self) -> list[dict[str, Any]]:
        return [x.to_public() for x in self._items]

    def get(self, sample_id: str) -> SampleItem:
        if sample_id not in self._by_id:
            raise KeyError(sample_id)
        return self._by_id[sample_id]

    def size(self) -> int:
        return len(self._items)
