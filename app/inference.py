from __future__ import annotations

import base64
import io
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

from app.config import PROJECT_ROOT


def pick_device(explicit: str) -> torch.device:
    if explicit != "auto":
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_artifact_path(path_value: str | Path, *, config_dir: Path | None = None) -> Path:
    raw = str(path_value).strip()
    candidate = Path(raw).expanduser()

    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute() and config_dir is not None:
        rel = (config_dir / candidate).resolve()
        if rel.exists():
            return rel

    normalized = raw.replace("\\", "/")
    for marker, shift in (("/data/", 1), ("data/", 0)):
        idx = normalized.find(marker)
        if idx >= 0:
            rel_part = normalized[idx + shift :]
            remapped = (PROJECT_ROOT / rel_part).resolve()
            if remapped.exists():
                return remapped

    return candidate.resolve()


def build_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model
    if name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model
    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


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


def read_image_bytes(file_bytes: bytes, filename: str) -> tuple[np.ndarray, str]:
    name = filename.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        mono1 = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
        x = _normalize_to_unit(arr, mono1=mono1)
        return x, "dicom"

    with Image.open(io.BytesIO(file_bytes)) as im:
        gray = im.convert("L")
        x = np.asarray(gray, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0), "image"


def preprocess_for_model(image: np.ndarray, image_size: int) -> torch.Tensor:
    t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).repeat(3, 1, 1)
    return t.clamp(0.0, 1.0).unsqueeze(0)


def encode_png_data_url(image_rgb_or_gray: np.ndarray) -> str:
    if image_rgb_or_gray.ndim == 2:
        pil_img = Image.fromarray(image_rgb_or_gray.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(image_rgb_or_gray.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def find_last_conv2d(model: nn.Module) -> nn.Module:
    last: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last


def grad_cam_single_model(model: nn.Module, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
    target_layer = find_last_conv2d(model)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _forward_hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        activations["value"] = out

    def _backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            gradients["value"] = grad_output[0]

    hook_fwd = target_layer.register_forward_hook(_forward_hook)
    hook_bwd = target_layer.register_full_backward_hook(_backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = torch.nan_to_num(model(input_tensor), nan=0.0, posinf=30.0, neginf=-30.0)
        score = logits[:, class_idx].sum()
        score.backward()
    finally:
        hook_fwd.remove()
        hook_bwd.remove()

    if "value" not in activations or "value" not in gradients:
        h, w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        return np.zeros((h, w), dtype=np.float32)

    acts = activations["value"].detach()
    grads = gradients["value"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
    cam_np -= float(cam_np.min())
    den = float(cam_np.max() - cam_np.min())
    if den > 1e-8:
        cam_np /= den
    return np.clip(cam_np, 0.0, 1.0)


def make_visuals(original_unit: np.ndarray, cam_unit: np.ndarray) -> dict[str, Any]:
    h, w = original_unit.shape[:2]
    cam_resized = cv2.resize(cam_unit, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_8 = np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    original_8 = np.clip(original_unit * 255.0, 0, 255).astype(np.uint8)
    original_rgb = np.stack([original_8, original_8, original_8], axis=-1)

    overlay_rgb = cv2.addWeighted(original_rgb, 0.45, heatmap_rgb, 0.55, 0.0)
    lesion_mask = (cam_8 >= 190).astype(np.uint8) * 255

    ys, xs = np.where(lesion_mask > 0)
    hotspot_area_pct = float((lesion_mask > 0).sum() / max(lesion_mask.size, 1) * 100.0)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
    else:
        bbox = None

    return {
        "original_png": encode_png_data_url(original_rgb),
        "heatmap_png": encode_png_data_url(heatmap_rgb),
        "overlay_png": encode_png_data_url(overlay_rgb),
        "mask_png": encode_png_data_url(lesion_mask),
        "bbox": bbox,
        "stats": {
            "hotspot_area_pct": hotspot_area_pct,
            "cam_peak": float(np.max(cam_resized)),
            "cam_mean": float(np.mean(cam_resized)),
        },
    }


@dataclass
class ModelBundle:
    name: str
    models: list[nn.Module]
    image_size: int
    threshold: float
    checkpoints: list[Path]
    ensemble_weights: list[float]
    tta_hflip: bool
    target_recall: float | None
    selection_summary: str | None
    note: str | None


class PredictorService:
    def __init__(self, device: str = "auto") -> None:
        self.device = pick_device(device)
        self._bundles: dict[str, ModelBundle] = {}

    def load_mode(self, mode_name: str, config_path: Path) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Model config does not exist: {config_path}")

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = str(cfg.get("model", "efficientnet_b0"))
        config_dir = config_path.parent.resolve()
        checkpoint_paths: list[Path] = []
        if isinstance(cfg.get("ensemble_checkpoints"), list) and cfg.get("ensemble_checkpoints"):
            checkpoint_paths = [
                resolve_artifact_path(str(p), config_dir=config_dir) for p in cfg["ensemble_checkpoints"]
            ]
        else:
            checkpoint_paths.append(resolve_artifact_path(str(cfg["checkpoint"]), config_dir=config_dir))
            if cfg.get("ensemble_with"):
                checkpoint_paths.append(resolve_artifact_path(str(cfg["ensemble_with"]), config_dir=config_dir))

        if not checkpoint_paths:
            raise ValueError("No checkpoints found in config.")
        for p in checkpoint_paths:
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint missing: {p}")

        models_list: list[nn.Module] = []
        for checkpoint in checkpoint_paths:
            model = build_model(model_name).to(self.device)
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models_list.append(model)

        cfg_weights = cfg.get("ensemble_weights")
        if isinstance(cfg_weights, list) and len(cfg_weights) == len(checkpoint_paths):
            weights = np.asarray([float(x) for x in cfg_weights], dtype=np.float64)
        else:
            weights = np.ones(len(checkpoint_paths), dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(checkpoint_paths), dtype=np.float64)
        weights = weights / float(weights.sum())

        bundle = ModelBundle(
            name=model_name,
            models=models_list,
            image_size=int(cfg.get("image_size", 384)),
            threshold=float(cfg.get("threshold", 0.5)),
            checkpoints=checkpoint_paths,
            ensemble_weights=[float(x) for x in weights.tolist()],
            tta_hflip=bool(cfg.get("tta_hflip", False)),
            target_recall=float(cfg.get("target_recall")) if cfg.get("target_recall") is not None else None,
            selection_summary=str(cfg.get("selection_summary")) if cfg.get("selection_summary") else None,
            note=str(cfg.get("note")) if cfg.get("note") else None,
        )
        self._bundles[mode_name] = bundle

    def available_modes(self) -> list[str]:
        return sorted(self._bundles.keys())

    def mode_profiles(self) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for mode_name, bundle in self._bundles.items():
            profiles[mode_name] = {
                "mode": mode_name,
                "model": bundle.name,
                "image_size": int(bundle.image_size),
                "threshold": float(bundle.threshold),
                "ensemble_size": int(len(bundle.models)),
                "tta_hflip": bool(bundle.tta_hflip),
                "target_recall": float(bundle.target_recall) if bundle.target_recall is not None else None,
                "selection_summary": bundle.selection_summary,
                "note": bundle.note,
            }
        return profiles

    def _model_probability(self, model: nn.Module, input_tensor: torch.Tensor, tta_hflip: bool) -> float:
        with torch.no_grad():
            logits = torch.nan_to_num(model(input_tensor), nan=0.0, posinf=30.0, neginf=-30.0)
            prob = torch.softmax(logits, dim=1)[:, 1]
            if tta_hflip:
                flipped = torch.flip(input_tensor, dims=[3])
                logits_f = torch.nan_to_num(model(flipped), nan=0.0, posinf=30.0, neginf=-30.0)
                prob_f = torch.softmax(logits_f, dim=1)[:, 1]
                prob = 0.5 * (prob + prob_f)
        return float(prob.item())

    def predict(self, *, file_bytes: bytes, filename: str, mode: str) -> dict[str, Any]:
        if mode not in self._bundles:
            raise ValueError(f"Unknown mode '{mode}'. Available: {self.available_modes()}")
        bundle = self._bundles[mode]

        t0 = time.time()
        unit_image, source_kind = read_image_bytes(file_bytes, filename=filename)
        original_height, original_width = int(unit_image.shape[0]), int(unit_image.shape[1])
        input_tensor = preprocess_for_model(unit_image, image_size=bundle.image_size).to(self.device)

        probs = [
            self._model_probability(model=m, input_tensor=input_tensor, tta_hflip=bundle.tta_hflip)
            for m in bundle.models
        ]
        weights = np.asarray(bundle.ensemble_weights, dtype=np.float64)
        malignant_probability = float(np.dot(weights, np.asarray(probs, dtype=np.float64)))
        used_checkpoints = [str(p) for p in bundle.checkpoints]
        threshold = float(bundle.threshold)
        predicted_label = "malignant" if malignant_probability >= threshold else "benign"
        confidence = malignant_probability if predicted_label == "malignant" else (1.0 - malignant_probability)
        delta_to_threshold = float(malignant_probability - threshold)
        confidence_band = (
            "high"
            if confidence >= 0.85
            else ("medium" if confidence >= 0.70 else "low")
        )

        cam_input = input_tensor.clone().detach().requires_grad_(True)
        cam_unit = grad_cam_single_model(bundle.models[0], cam_input, class_idx=1)
        visuals = make_visuals(unit_image, cam_unit)
        hotspot_area_pct = float(visuals.get("stats", {}).get("hotspot_area_pct", 0.0))

        warnings: list[str] = []
        if source_kind == "dicom" and max(original_width, original_height) >= 1800:
            warnings.append(
                "Input appears to be a full-field mammogram. Current classifier was trained on lesion-centered crops; "
                "for full-image uploads, localization may be less stable."
            )
        if hotspot_area_pct < 0.2:
            warnings.append(
                "Very small activation region detected (<0.2% of image area). Treat this prediction as low-interpretability."
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        return {
            "prediction_id": str(uuid.uuid4()),
            "mode": mode,
            "source_kind": source_kind,
            "file_name": filename,
            "image": {
                "width": original_width,
                "height": original_height,
            },
            "model": {
                "name": bundle.name,
                "image_size": bundle.image_size,
                "threshold": threshold,
                "checkpoints": used_checkpoints,
                "ensemble_size": int(len(bundle.models)),
                "ensemble_weights": [float(x) for x in bundle.ensemble_weights],
                "tta_hflip": bundle.tta_hflip,
            },
            "prediction": {
                "label": predicted_label,
                "malignant_probability": malignant_probability,
                "confidence": float(confidence),
            },
            "decision": {
                "threshold": threshold,
                "delta_to_threshold": delta_to_threshold,
                "confidence_band": confidence_band,
            },
            "explainability": {
                "bbox": visuals["bbox"],
                "stats": visuals["stats"],
                "images": {
                    "original": visuals["original_png"],
                    "heatmap": visuals["heatmap_png"],
                    "overlay": visuals["overlay_png"],
                    "mask": visuals["mask_png"],
                },
            },
            "warnings": warnings,
            "timing_ms": float(elapsed_ms),
        }
