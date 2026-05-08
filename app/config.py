from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STANDARD_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "models"
    / "classifier_v2_seed7_qc_champion"
    / "inference_config_balanced_ensemble.json"
)
DEFAULT_HIGH_RECALL_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "models"
    / "classifier_v2_seed7_qc_champion"
    / "inference_config_high_recall.json"
)
DEFAULT_SAMPLE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "classification_manifest_full_cropped.csv"


@dataclass(frozen=True)
class AppSettings:
    host: str
    port: int
    reload: bool
    standard_config: Path
    high_recall_config: Path
    sample_manifest: Path
    sample_count: int
    device: str

    @staticmethod
    def from_env() -> "AppSettings":
        return AppSettings(
            host=os.getenv("CANCER_API_HOST", "0.0.0.0"),
            port=int(os.getenv("CANCER_API_PORT", "8005")),
            reload=os.getenv("CANCER_API_RELOAD", "0").strip() in {"1", "true", "True"},
            standard_config=Path(os.getenv("CANCER_MODEL_CONFIG", str(DEFAULT_STANDARD_CONFIG))).expanduser().resolve(),
            high_recall_config=Path(
                os.getenv("CANCER_MODEL_CONFIG_HIGH_RECALL", str(DEFAULT_HIGH_RECALL_CONFIG))
            ).expanduser().resolve(),
            sample_manifest=Path(
                os.getenv("CANCER_SAMPLE_MANIFEST", str(DEFAULT_SAMPLE_MANIFEST))
            ).expanduser().resolve(),
            sample_count=int(os.getenv("CANCER_SAMPLE_COUNT", "20")),
            device=os.getenv("CANCER_DEVICE", "auto"),
        )
