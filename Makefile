PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PY_RUN := $(if $(wildcard $(PY)),$(PY),$(PYTHON))

.PHONY: venv install serve-api serve-api-v4 serve-api-champion data-dirs data-tcia-layout data-verify data-index data-mini data-report data-split-mini data-manifest-mini-cls data-manifest-full-cls data-manifest-full-cls-cropped data-manifest-full-cls-cropped-qc train-cls-mini train-cls-mini-resume train-cls-full train-cls-full-resume train-cls-full-sota train-cls-full-sota-resume train-cls-full-sota-bg train-cls-full-sota-bg-resume train-cls-full-sota-v2 train-cls-full-sota-v2-resume train-cls-full-sota-v2-qc train-cls-full-sota-v2-qc-resume train-cls-full-sota-v2-qc-select train-cls-full-sota-select train-cls-full-sota-v2-select train-cls-full-sota-v2-select-all eval-cls-full-sota-v2-ensemble train-cls-full-sota-v3-ft train-cls-full-sota-v3-ft-resume train-cls-full-sota-v3-ft-bg train-cls-full-sota-v3-ft-bg-resume train-cls-full-sota-v3-ft-select train-cls-full-sota-v4 train-cls-full-sota-v4-resume train-cls-full-sota-v4-bg train-cls-full-sota-v4-bg-resume train-cls-full-sota-v4-stop train-cls-full-sota-v4-select eval-cls-full-sota-v4-topk-ensemble build-inference-config-v4 train-cls-full-sota-v5-b0-ft train-cls-full-sota-v5-b0-ft-resume train-cls-full-sota-v5-b0-ft-bg train-cls-full-sota-v5-b0-ft-bg-resume train-cls-full-sota-v5-b0-ft-select eval-cls-full-sota-v5-b0-ft-ensemble build-inference-config-v5-b0 train-cls-full-sota-v5-b2-gentle train-cls-full-sota-v5-b2-gentle-resume train-cls-full-sota-v5-b2-gentle-bg train-cls-full-sota-v5-b2-gentle-bg-resume train-cls-full-sota-v5-b2-gentle-select eval-cls-full-sota-v5-b2-gentle-ensemble build-inference-config-v5-b2 train-demo train-demo-resume

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

serve-api:
	$(PY_RUN) -m uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload

serve-api-v4:
	CANCER_MODEL_CONFIG=/Users/dmitriderevjanko/Desktop/cancer_detector/data/artifacts/models/classifier_v4_champion/inference_config.json \
	CANCER_MODEL_CONFIG_HIGH_RECALL=/Users/dmitriderevjanko/Desktop/cancer_detector/data/artifacts/models/classifier_v4_champion/inference_config_high_recall.json \
	$(PY_RUN) -m uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload

serve-api-champion:
	CANCER_MODEL_CONFIG=/Users/dmitriderevjanko/Desktop/cancer_detector/data/artifacts/models/classifier_v2_seed7_qc_champion/inference_config_balanced_ensemble.json \
	CANCER_MODEL_CONFIG_HIGH_RECALL=/Users/dmitriderevjanko/Desktop/cancer_detector/data/artifacts/models/classifier_v2_seed7_qc_champion/inference_config_high_recall.json \
	$(PY_RUN) -m uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload

data-dirs:
	mkdir -p data/raw data/interim data/processed data/artifacts

data-tcia-layout: data-dirs
	mkdir -p data/raw/cbis_ddsm_tcia/dicom
	mkdir -p data/raw/cbis_ddsm_tcia/csv
	mkdir -p data/raw/cbis_ddsm_tcia/index

data-verify: data-tcia-layout
	$(PY_RUN) scripts/verify_tcia_download.py --root data/raw/cbis_ddsm_tcia

data-index: data-tcia-layout
	$(PY_RUN) scripts/index_dicom_headers.py \
		--root data/raw/cbis_ddsm_tcia/dicom \
		--out data/raw/cbis_ddsm_tcia/index/dicom_index.csv

data-mini: data-dirs
	$(PY_RUN) scripts/build_mini_dataset.py \
		--source-root data/raw/cbis_ddsm_tcia \
		--out-root data/interim/cbis_ddsm_mini \
		--per-class 80 \
		--seed 42 \
		--link-mode symlink

data-report:
	$(PY_RUN) scripts/report_dataset.py --root data/interim/cbis_ddsm_mini

data-split-mini:
	$(PY_RUN) scripts/create_splits.py \
		--manifest data/interim/cbis_ddsm_mini/subset_manifest.csv \
		--out data/interim/cbis_ddsm_mini/subset_manifest_splits.csv \
		--seed 42 \
		--test-size 0.15 \
		--val-size 0.15

data-manifest-mini-cls:
	$(PY_RUN) scripts/build_mini_classification_manifest.py \
		--subset-manifest data/interim/cbis_ddsm_mini/subset_manifest_splits.csv \
		--index-csv data/raw/cbis_ddsm_tcia/index/dicom_index.csv \
		--out data/processed/classification_manifest_mini.csv

data-manifest-full-cls:
	$(PY_RUN) scripts/build_classification_manifest.py \
		--source-root data/raw/cbis_ddsm_tcia \
		--path-column "image file path" \
		--val-size 0.15 \
		--seed 42 \
		--out data/processed/classification_manifest_full.csv

data-manifest-full-cls-cropped:
	$(PY_RUN) scripts/build_classification_manifest.py \
		--source-root data/raw/cbis_ddsm_tcia \
		--path-column "cropped image file path" \
		--val-size 0.15 \
		--seed 42 \
		--out data/processed/classification_manifest_full_cropped.csv

data-manifest-full-cls-cropped-qc:
	$(PY_RUN) scripts/filter_classification_manifest_quality.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--out data/processed/classification_manifest_full_cropped_qc.csv \
		--stats-out data/processed/classification_manifest_full_cropped_qc_stats.csv \
		--dropped-out data/processed/classification_manifest_full_cropped_qc_dropped.csv \
		--min-tissue-ratio 0.001 \
		--min-std 0.02 \
		--min-dyn-p99-p1 0.0 \
		--train-quantile 0.05

train-cls-mini:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_mini.csv \
		--run-dir data/artifacts/runs/classifier_mini_v1 \
		--model resnet18 \
		--epochs 30 \
		--batch-size 8 \
		--image-size 512 \
		--lr 3e-4 \
		--weight-decay 1e-4 \
		--threshold 0.5 \
		--save-every 2 \
		--early-stop-patience 8 \
		--num-workers 0 \
		--cache-dir data/artifacts/cache/classifier_mini_v1

train-cls-mini-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_mini.csv \
		--run-dir data/artifacts/runs/classifier_mini_v1 \
		--model resnet18 \
		--epochs 30 \
		--batch-size 8 \
		--image-size 512 \
		--lr 3e-4 \
		--weight-decay 1e-4 \
		--threshold 0.5 \
		--save-every 2 \
		--early-stop-patience 8 \
		--num-workers 0 \
		--cache-dir data/artifacts/cache/classifier_mini_v1 \
		--resume

train-cls-full:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full.csv \
		--run-dir data/artifacts/runs/classifier_full_v1 \
		--model resnet18 \
		--epochs 80 \
		--batch-size 4 \
		--image-size 512 \
		--lr 3e-4 \
		--weight-decay 1e-4 \
		--threshold 0.5 \
		--save-every 2 \
		--early-stop-patience 12 \
		--num-workers 0 \
		--cache-dir data/artifacts/cache/classifier_full_v1

train-cls-full-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full.csv \
		--run-dir data/artifacts/runs/classifier_full_v1 \
		--model resnet18 \
		--epochs 80 \
		--batch-size 4 \
		--image-size 512 \
		--lr 3e-4 \
		--weight-decay 1e-4 \
		--threshold 0.5 \
		--save-every 2 \
		--early-stop-patience 12 \
		--num-workers 0 \
		--cache-dir data/artifacts/cache/classifier_full_v1 \
		--resume

train-cls-full-sota:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v1 \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 140 \
		--batch-size 8 \
		--image-size 384 \
		--lr 2e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.85 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 25 \
		--num-workers 0 \
		--balanced-sampler \
		--cache-dir data/artifacts/cache/classifier_full_sota_v1

train-cls-full-sota-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v1 \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 140 \
		--batch-size 8 \
		--image-size 384 \
		--lr 2e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.85 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 25 \
		--num-workers 0 \
		--balanced-sampler \
		--cache-dir data/artifacts/cache/classifier_full_sota_v1 \
		--resume

train-cls-full-sota-bg:
	mkdir -p data/artifacts/runs/classifier_full_sota_v1
	nohup caffeinate -dimsu $(MAKE) train-cls-full-sota > data/artifacts/runs/classifier_full_sota_v1/nohup.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v1/pid.txt

train-cls-full-sota-bg-resume:
	mkdir -p data/artifacts/runs/classifier_full_sota_v1
	nohup caffeinate -dimsu $(MAKE) train-cls-full-sota-resume > data/artifacts/runs/classifier_full_sota_v1/nohup_resume.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v1/pid_resume.txt

train-cls-full-sota-v2:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2 \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 180 \
		--batch-size 8 \
		--image-size 384 \
		--lr 1.5e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 35 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2

train-cls-full-sota-v2-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2 \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 180 \
		--batch-size 8 \
		--image-size 384 \
		--lr 1.5e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 35 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2 \
		--resume

train-cls-full-sota-v2-qc:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped_qc.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2_qc \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 180 \
		--batch-size 8 \
		--image-size 384 \
		--lr 1.5e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 35 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_qc

train-cls-full-sota-v2-qc-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped_qc.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2_qc \
		--model efficientnet_b0 \
		--pretrained \
		--epochs 180 \
		--batch-size 8 \
		--image-size 384 \
		--lr 1.5e-4 \
		--weight-decay 1e-4 \
		--threshold 0.35 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 35 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_qc \
		--resume

train-cls-full-sota-v2-qc-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped_qc.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2_qc \
		--model efficientnet_b0 \
		--image-size 384 \
		--batch-size 8 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.75 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--top-k-errors 30 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_qc_eval

train-cls-full-sota-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v1 \
		--model efficientnet_b0 \
		--image-size 384 \
		--batch-size 8 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.80 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--top-k-errors 30 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v1_eval

train-cls-full-sota-v2-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2 \
		--model efficientnet_b0 \
		--image-size 384 \
		--batch-size 8 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.75 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--top-k-errors 30 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_eval

train-cls-full-sota-v2-select-all:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2 \
		--model efficientnet_b0 \
		--image-size 384 \
		--batch-size 8 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--include-epoch-checkpoints \
		--max-checkpoints 80 \
		--top-k-errors 40 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_eval_all_epochs

eval-cls-full-sota-v2-ensemble:
	$(PY_RUN) scripts/evaluate_classifier_ensemble.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v2 \
		--checkpoint-names best_auc.pt,last.pt \
		--model efficientnet_b0 \
		--image-size 384 \
		--batch-size 8 \
		--num-workers 0 \
		--tta-hflip \
		--threshold-mode recall_floor \
		--target-recall 0.80 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v2_ensemble_eval

train-cls-full-sota-v3-ft:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v3_ft \
		--model efficientnet_b0 \
		--epochs 80 \
		--batch-size 6 \
		--image-size 448 \
		--lr 6e-5 \
		--weight-decay 1e-4 \
		--threshold 0.06 \
		--threshold-mode val_recall_floor \
		--target-recall 0.80 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 20 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--init-checkpoint data/artifacts/runs/classifier_full_sota_v2/checkpoints/best_auc.pt \
		--cache-dir data/artifacts/cache/classifier_full_sota_v3_ft

train-cls-full-sota-v3-ft-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v3_ft \
		--model efficientnet_b0 \
		--epochs 80 \
		--batch-size 6 \
		--image-size 448 \
		--lr 6e-5 \
		--weight-decay 1e-4 \
		--threshold 0.06 \
		--threshold-mode val_recall_floor \
		--target-recall 0.80 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 20 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v3_ft \
		--resume

train-cls-full-sota-v3-ft-bg:
	mkdir -p data/artifacts/runs/classifier_full_sota_v3_ft
	nohup caffeinate -dimsu $(MAKE) train-cls-full-sota-v3-ft > data/artifacts/runs/classifier_full_sota_v3_ft/nohup.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v3_ft/pid.txt

train-cls-full-sota-v3-ft-bg-resume:
	mkdir -p data/artifacts/runs/classifier_full_sota_v3_ft
	nohup caffeinate -dimsu $(MAKE) train-cls-full-sota-v3-ft-resume > data/artifacts/runs/classifier_full_sota_v3_ft/nohup_resume.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v3_ft/pid_resume.txt

train-cls-full-sota-v3-ft-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v3_ft \
		--model efficientnet_b0 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.80 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--top-k-errors 30 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v3_ft_eval

train-cls-full-sota-v4:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v4 \
		--model efficientnet_b2 \
		--pretrained \
		--epochs 260 \
		--batch-size 6 \
		--image-size 448 \
		--lr 1.2e-4 \
		--weight-decay 5e-5 \
		--loss focal \
		--focal-gamma 1.75 \
		--label-smoothing 0.02 \
		--mixup-alpha 0.15 \
		--aug-profile strong \
		--freeze-backbone-epochs 3 \
		--threshold 0.10 \
		--threshold-mode val_recall_floor \
		--target-recall 0.85 \
		--threshold-min 0.03 \
		--threshold-max 0.95 \
		--threshold-steps 93 \
		--save-every 2 \
		--early-stop-patience 45 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v4

train-cls-full-sota-v4-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v4 \
		--model efficientnet_b2 \
		--pretrained \
		--epochs 260 \
		--batch-size 6 \
		--image-size 448 \
		--lr 1.2e-4 \
		--weight-decay 5e-5 \
		--loss focal \
		--focal-gamma 1.75 \
		--label-smoothing 0.02 \
		--mixup-alpha 0.15 \
		--aug-profile strong \
		--freeze-backbone-epochs 3 \
		--threshold 0.10 \
		--threshold-mode val_recall_floor \
		--target-recall 0.85 \
		--threshold-min 0.03 \
		--threshold-max 0.95 \
		--threshold-steps 93 \
		--save-every 2 \
		--early-stop-patience 45 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v4 \
		--resume

train-cls-full-sota-v4-bg:
	mkdir -p data/artifacts/runs/classifier_full_sota_v4
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v4 > data/artifacts/runs/classifier_full_sota_v4/nohup.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v4/pid.txt

train-cls-full-sota-v4-bg-resume:
	mkdir -p data/artifacts/runs/classifier_full_sota_v4
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v4-resume > data/artifacts/runs/classifier_full_sota_v4/nohup_resume.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v4/pid_resume.txt

train-cls-full-sota-v4-stop:
	@if [ -f data/artifacts/runs/classifier_full_sota_v4/pid.txt ]; then \
		PID=$$(cat data/artifacts/runs/classifier_full_sota_v4/pid.txt); \
		echo "Stopping background run PID=$$PID"; \
		kill -TERM $$PID 2>/dev/null || true; \
	else \
		echo "No pid.txt found for v4 run."; \
	fi
	@pkill -f "scripts/train_classifier.py --manifest data/processed/classification_manifest_full_cropped.csv --run-dir data/artifacts/runs/classifier_full_sota_v4" 2>/dev/null || true

train-cls-full-sota-v4-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v4 \
		--model efficientnet_b2 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.03 \
		--threshold-max 0.95 \
		--threshold-steps 93 \
		--include-epoch-checkpoints \
		--max-checkpoints 60 \
		--top-k-errors 40 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v4_eval

eval-cls-full-sota-v4-topk-ensemble:
	$(PY_RUN) scripts/evaluate_classifier_ensemble.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v4 \
		--ranking-csv data/artifacts/runs/classifier_full_sota_v4/analysis/checkpoint_selection/checkpoint_ranking.csv \
		--top-k-from-ranking 5 \
		--model efficientnet_b2 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--tta-hflip \
		--ensemble-weighting val_auc \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.03 \
		--threshold-max 0.95 \
		--threshold-steps 93 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v4_ensemble_eval

build-inference-config-v4:
	$(PY_RUN) scripts/build_inference_configs.py \
		--balanced-summary data/artifacts/runs/classifier_full_sota_v4/analysis/checkpoint_selection/summary.json \
		--ensemble-summary data/artifacts/runs/classifier_full_sota_v4/analysis/ensemble_eval/summary.json \
		--model efficientnet_b2 \
		--image-size 448 \
		--out-dir data/artifacts/models/classifier_v4_champion

train-cls-full-sota-v5-b0-ft:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b0_ft \
		--model efficientnet_b0 \
		--epochs 70 \
		--batch-size 6 \
		--image-size 448 \
		--lr 3e-5 \
		--weight-decay 1e-4 \
		--threshold 0.06 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 18 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--init-checkpoint data/artifacts/runs/classifier_full_sota_v2/checkpoints/best_auc.pt \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b0_ft

train-cls-full-sota-v5-b0-ft-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b0_ft \
		--model efficientnet_b0 \
		--epochs 70 \
		--batch-size 6 \
		--image-size 448 \
		--lr 3e-5 \
		--weight-decay 1e-4 \
		--threshold 0.06 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 18 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b0_ft \
		--resume

train-cls-full-sota-v5-b0-ft-bg:
	mkdir -p data/artifacts/runs/classifier_full_sota_v5_b0_ft
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v5-b0-ft > data/artifacts/runs/classifier_full_sota_v5_b0_ft/nohup.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v5_b0_ft/pid.txt

train-cls-full-sota-v5-b0-ft-bg-resume:
	mkdir -p data/artifacts/runs/classifier_full_sota_v5_b0_ft
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v5-b0-ft-resume > data/artifacts/runs/classifier_full_sota_v5_b0_ft/nohup_resume.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v5_b0_ft/pid_resume.txt

train-cls-full-sota-v5-b0-ft-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b0_ft \
		--model efficientnet_b0 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--include-epoch-checkpoints \
		--max-checkpoints 40 \
		--top-k-errors 40 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b0_ft_eval

eval-cls-full-sota-v5-b0-ft-ensemble:
	$(PY_RUN) scripts/evaluate_classifier_ensemble.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b0_ft \
		--checkpoint-names best_auc.pt,last.pt \
		--model efficientnet_b0 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--tta-hflip \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b0_ft_ensemble_eval

build-inference-config-v5-b0:
	$(PY_RUN) scripts/build_inference_configs.py \
		--balanced-summary data/artifacts/runs/classifier_full_sota_v5_b0_ft/analysis/checkpoint_selection/summary.json \
		--ensemble-summary data/artifacts/runs/classifier_full_sota_v5_b0_ft/analysis/ensemble_eval/summary.json \
		--model efficientnet_b0 \
		--image-size 448 \
		--out-dir data/artifacts/models/classifier_v5_b0_champion

train-cls-full-sota-v5-b2-gentle:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b2_gentle \
		--model efficientnet_b2 \
		--pretrained \
		--epochs 90 \
		--batch-size 6 \
		--image-size 448 \
		--lr 5e-5 \
		--weight-decay 1e-4 \
		--aug-profile basic \
		--freeze-backbone-epochs 3 \
		--threshold 0.08 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 22 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b2_gentle

train-cls-full-sota-v5-b2-gentle-resume:
	$(PY_RUN) scripts/train_classifier.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b2_gentle \
		--model efficientnet_b2 \
		--pretrained \
		--epochs 90 \
		--batch-size 6 \
		--image-size 448 \
		--lr 5e-5 \
		--weight-decay 1e-4 \
		--aug-profile basic \
		--freeze-backbone-epochs 3 \
		--threshold 0.08 \
		--threshold-mode val_recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--save-every 2 \
		--early-stop-patience 22 \
		--num-workers 0 \
		--balanced-sampler \
		--disable-class-weights \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b2_gentle \
		--resume

train-cls-full-sota-v5-b2-gentle-bg:
	mkdir -p data/artifacts/runs/classifier_full_sota_v5_b2_gentle
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v5-b2-gentle > data/artifacts/runs/classifier_full_sota_v5_b2_gentle/nohup.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v5_b2_gentle/pid.txt

train-cls-full-sota-v5-b2-gentle-bg-resume:
	mkdir -p data/artifacts/runs/classifier_full_sota_v5_b2_gentle
	nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu $(MAKE) train-cls-full-sota-v5-b2-gentle-resume > data/artifacts/runs/classifier_full_sota_v5_b2_gentle/nohup_resume.log 2>&1 & echo $$! > data/artifacts/runs/classifier_full_sota_v5_b2_gentle/pid_resume.txt

train-cls-full-sota-v5-b2-gentle-select:
	$(PY_RUN) scripts/select_classifier_checkpoint.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b2_gentle \
		--model efficientnet_b2 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--include-epoch-checkpoints \
		--max-checkpoints 45 \
		--top-k-errors 40 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b2_gentle_eval

eval-cls-full-sota-v5-b2-gentle-ensemble:
	$(PY_RUN) scripts/evaluate_classifier_ensemble.py \
		--manifest data/processed/classification_manifest_full_cropped.csv \
		--run-dir data/artifacts/runs/classifier_full_sota_v5_b2_gentle \
		--checkpoint-names best_auc.pt,last.pt \
		--model efficientnet_b2 \
		--image-size 448 \
		--batch-size 6 \
		--num-workers 0 \
		--tta-hflip \
		--threshold-mode recall_floor \
		--target-recall 0.82 \
		--threshold-min 0.05 \
		--threshold-max 0.90 \
		--threshold-steps 86 \
		--cache-dir data/artifacts/cache/classifier_full_sota_v5_b2_gentle_ensemble_eval

build-inference-config-v5-b2:
	$(PY_RUN) scripts/build_inference_configs.py \
		--balanced-summary data/artifacts/runs/classifier_full_sota_v5_b2_gentle/analysis/checkpoint_selection/summary.json \
		--ensemble-summary data/artifacts/runs/classifier_full_sota_v5_b2_gentle/analysis/ensemble_eval/summary.json \
		--model efficientnet_b2 \
		--image-size 448 \
		--out-dir data/artifacts/models/classifier_v5_b2_champion

train-demo:
	$(PY_RUN) scripts/train_resume_demo.py \
		--run-dir data/artifacts/runs/demo_classifier_v1 \
		--epochs 30 \
		--batch-size 64 \
		--lr 1e-3

train-demo-resume:
	$(PY_RUN) scripts/train_resume_demo.py \
		--run-dir data/artifacts/runs/demo_classifier_v1 \
		--epochs 30 \
		--batch-size 64 \
		--lr 1e-3 \
		--resume
