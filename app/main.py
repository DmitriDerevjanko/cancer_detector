from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import AppSettings, PROJECT_ROOT
from app.inference import PredictorService
from app.samples import SampleCatalog


settings = AppSettings.from_env()
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Breast Cancer Clinical Decision Support API",
    version="0.2.0",
    description="Research-grade inference service with explainability overlays.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

service = PredictorService(device=settings.device)
sample_catalog = SampleCatalog()
sample_catalog_error: str | None = None


@app.on_event("startup")
def startup_load_models() -> None:
    global sample_catalog_error
    service.load_mode("balanced", settings.standard_config)
    if settings.high_recall_config.exists():
        service.load_mode("high_recall", settings.high_recall_config)
    try:
        sample_catalog.load_from_manifest(settings.sample_manifest, sample_count=settings.sample_count, seed=42)
        sample_catalog_error = None
        logger.info("Loaded demo samples: %s", sample_catalog.size())
    except Exception as exc:  # noqa: BLE001
        manifest_error = str(exc)
        try:
            sample_catalog.load_from_directory(
                PROJECT_ROOT / "data" / "artifacts" / "demo_samples",
                sample_count=settings.sample_count,
            )
            sample_catalog_error = None
            logger.info("Loaded curated demo samples: %s", sample_catalog.size())
        except Exception as fallback_exc:  # noqa: BLE001
            sample_catalog_error = f"{manifest_error}; fallback failed: {fallback_exc}"
            logger.warning("Demo sample catalog unavailable: %s", sample_catalog_error)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "modes": service.available_modes()}


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    modes = service.available_modes()
    default_mode = "balanced" if "balanced" in modes else (modes[0] if modes else "")
    return {
        "available_modes": modes,
        "default_mode": default_mode,
        "mode_profiles": service.mode_profiles(),
        "research_warning": "This system is for research purposes only and not intended for clinical use.",
    }


@app.get("/api/samples")
def list_samples() -> dict[str, Any]:
    items = sample_catalog.list_public()
    return {
        "ready": len(items) > 0,
        "count": len(items),
        "requested_count": settings.sample_count,
        "error": sample_catalog_error,
        "items": items,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("balanced")) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is missing.")
    if file.size is not None and file.size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        payload = await file.read()
        result = service.predict(file_bytes=payload, filename=file.filename, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        await file.close()
    return result


@app.post("/api/predict/sample/{sample_id}")
def predict_sample(sample_id: str, mode: str = "balanced") -> dict[str, Any]:
    if sample_catalog.size() == 0:
        raise HTTPException(status_code=503, detail="Sample gallery is not ready.")

    try:
        sample = sample_catalog.get(sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}") from exc

    try:
        payload = sample.dcm_path.read_bytes()
        result = service.predict(file_bytes=payload, filename=sample.file_name, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result["sample"] = {
        "sample_id": sample.sample_id,
        "ground_truth": sample.ground_truth,
        "source_split": sample.source_split,
        "file_name": sample.file_name,
    }
    return result


@app.get("/")
def index() -> FileResponse:
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    ico_path = frontend_dir / "favicon.ico"
    svg_path = frontend_dir / "favicon.svg"
    if ico_path.exists():
        return FileResponse(ico_path)
    if svg_path.exists():
        return FileResponse(svg_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="No favicon")
