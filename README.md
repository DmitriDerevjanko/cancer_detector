# OncoScope AI

OncoScope AI is an interactive research demo for breast imaging triage with explainability overlays.

Important: this project is for research/demo use only. It is not a medical device and must not be used for standalone clinical decisions.

## What This Demo Shows
- Fast local inference for mammography-style inputs (`.dcm`, `.png`, `.jpg`, `.jpeg`)
- Two operating modes:
  - `balanced`: better specificity/precision while preserving useful sensitivity
  - `high_recall`: higher sensitivity with expected false-positive increase
- Visual explainability output:
  - original view
  - Grad-CAM heatmap
  - overlay
  - lesion mask
- Safety-oriented UI:
  - confidence band
  - decision margin vs threshold
  - warnings and model limits panel

## Current Champion Demo Stack
The default local demo uses a 6-checkpoint ensemble (`efficientnet_b0` + TTA).

Primary configs:
- `data/artifacts/models/classifier_v2_seed7_qc_champion/inference_config_balanced_ensemble.json`
- `data/artifacts/models/classifier_v2_seed7_qc_champion/inference_config_high_recall.json`

Reference evaluation summaries:
- `data/artifacts/runs/ensemble_v2_seed7_qc_eval/summary.json`
- `data/artifacts/runs/ensemble_v2_seed7_qc_eval_r090/summary.json`

## Quick Start (Local)
```bash
cd /Users/dmitriderevjanko/Desktop/cancer_detector
make install
make serve-api-champion
```

Open:
- UI: `http://localhost:8005`
- API docs: `http://localhost:8005/docs`
- Health: `http://localhost:8005/health`

Note: first startup can take 1-2 minutes because multiple checkpoints and demo sample catalog are loaded.

## How To Test The Demo
1. Open `http://localhost:8005`.
2. Pick one of the 20 demo cases or upload a file.
3. Run `Balanced` mode and inspect:
   - malignancy probability
   - threshold
   - decision margin
   - confidence band
4. Switch to `High Recall` mode and compare behavior.
5. Verify that overlays (heatmap/mask) and warnings match expectations.

## API Endpoints
- `GET /health`
- `GET /api/model/info`
- `GET /api/samples`
- `POST /api/predict`
- `POST /api/predict/sample/{sample_id}`

## Project Layout
```text
app/
  config.py
  inference.py
  main.py
  samples.py
frontend/
  index.html
  styles.css
  app.js
scripts/
  train_classifier.py
  evaluate_classifier_ensemble.py
  select_classifier_checkpoint.py
  build_inference_configs.py
  filter_classification_manifest_quality.py
  ...
data/
  raw/
  interim/
  processed/
  artifacts/
```

## Training and Evaluation Commands
Base training/eval flow used in this project:
```bash
make train-cls-full-sota-v2
make train-cls-full-sota-v2-resume
make eval-cls-full-sota-v2-ensemble
make data-manifest-full-cls-cropped-qc
make train-cls-full-sota-v2-qc
make train-cls-full-sota-v2-qc-resume
make train-cls-full-sota-v2-qc-select
```

## Repository Hygiene (.gitignore Policy)
This repository is configured to avoid committing heavy raw/training artifacts by default.

Kept intentionally:
- curated demo image set (`data/artifacts/demo_samples/s001.png` ... `s020.png`)
- champion inference configs
- top checkpoints used by the champion ensemble
- key evaluation summaries

Ignored by default:
- raw datasets
- interim/processed full data dumps
- most training runs/checkpoints/caches

## Troubleshooting
If startup is slow:
- wait longer on first run (model loading can be heavy)
- check logs in the terminal running `make serve-api-champion`

If port `8005` is busy:
```bash
CANCER_API_PORT=8010 make serve-api-champion
```

If sample gallery is unavailable:
- ensure local DICOM paths from the selected manifest exist
- or upload images manually via the UI

## Deploy (VPS + Local Tunnel)
For production-demo deployment where the VPS only exposes domain/TLS and the backend stays on your local server, use:

- `deploy/README.md`
- `deploy/local/systemd/oncoview-api.service`
- `deploy/local/systemd/oncoview-reverse-tunnel.service`
- `deploy/vps/nginx/oncoview.conf`
- `.github/workflows/deploy-through-tunnel.yml`

## Compliance Notice
This project is a research prototype. It does not provide diagnosis, treatment recommendation, or clinical-grade risk estimation.
