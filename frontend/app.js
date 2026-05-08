import * as THREE from "https://unpkg.com/three@0.165.0/build/three.module.js";

const state = {
  mode: "balanced",
  file: null,
  selectedSampleId: null,
  busy: false,
  availableModes: ["balanced", "high_recall"],
  modeProfiles: {},
  samples: [],
};

const elements = {
  fileInput: document.getElementById("file-input"),
  fileName: document.getElementById("file-name"),
  sampleGrid: document.getElementById("sample-grid"),
  samplesMeta: document.getElementById("samples-meta"),
  modeButtons: Array.from(document.querySelectorAll(".mode-btn")),
  modeHelp: document.getElementById("mode-help"),
  predictButton: document.getElementById("predict-btn"),
  statusBox: document.getElementById("status-box"),
  modeLabel: document.getElementById("active-mode-label"),
  predictionLabel: document.getElementById("prediction-label"),
  confidenceLabel: document.getElementById("prediction-confidence"),
  resultMeta: document.getElementById("result-meta"),
  summaryLabel: document.getElementById("summary-label"),
  summaryProbability: document.getElementById("summary-probability"),
  summaryThreshold: document.getElementById("summary-threshold"),
  summaryConfidenceBand: document.getElementById("summary-confidence-band"),
  summaryBanner: document.getElementById("summary-banner"),
  decisionMargin: document.getElementById("decision-margin"),
  confidenceBand: document.getElementById("confidence-band"),
  hotspotArea: document.getElementById("hotspot-area"),
  warningText: document.getElementById("warning-text"),
  overlayStage: document.getElementById("overlay-stage"),
  imgOriginal: document.getElementById("img-original"),
  imgOverlay: document.getElementById("img-overlay"),
  imgHeatmap: document.getElementById("img-heatmap"),
  imgMask: document.getElementById("img-mask"),
  overlayAlpha: document.getElementById("overlay-alpha"),
  dropzone: document.getElementById("dropzone"),
  limitsList: document.getElementById("limits-list"),
};

const MODE_COPY = {
  balanced: "Balanced mode aims for better specificity/precision while preserving useful sensitivity.",
  high_recall: "High Recall mode is tuned to miss fewer positives, with expected increase in false positives.",
};

const BASE_LIMITS = [
  "Research-only system. Do not use as a standalone medical decision tool.",
  "Model is trained on lesion-centered crops; full-field uploads may be less stable.",
  "Predictions must be interpreted with radiologist review and clinical context.",
];

function hasInput() {
  return Boolean(state.file || state.selectedSampleId);
}

function setStatus(kind, text) {
  elements.statusBox.className = `status-box ${kind}`;
  elements.statusBox.textContent = text;
}

function setBusy(value) {
  state.busy = value;
  elements.predictButton.disabled = value || !hasInput();
  elements.predictButton.textContent = value ? "Analyzing..." : "Run Analysis";
}

function activateMode(mode) {
  state.mode = mode;
  elements.modeButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
    btn.disabled = !state.availableModes.includes(btn.dataset.mode);
  });
  elements.modeLabel.textContent = mode.replace("_", " ");
  const profile = state.modeProfiles[mode] || {};
  const baseHelp = profile.note || MODE_COPY[mode] || "Mode description is unavailable.";
  const thresholdText =
    typeof profile.threshold === "number" ? `thr=${Number(profile.threshold).toFixed(3)}` : "thr=—";
  const ensembleText =
    typeof profile.ensemble_size === "number" ? `ensemble=${profile.ensemble_size}` : "ensemble=—";
  const targetRecallText =
    typeof profile.target_recall === "number" ? `target recall=${(profile.target_recall * 100).toFixed(0)}%` : "";
  elements.modeHelp.textContent = [baseHelp, thresholdText, ensembleText, targetRecallText].filter(Boolean).join(" • ");
  renderLimits();
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderLimits(extra = []) {
  const modeSpecific =
    state.mode === "high_recall"
      ? "High Recall mode prioritizes sensitivity and may increase false positives."
      : "Balanced mode aims for stronger specificity while preserving useful sensitivity.";
  const lines = [...BASE_LIMITS, modeSpecific, ...extra];
  elements.limitsList.innerHTML = lines.map((line) => `<li>${line}</li>`).join("");
}

function setDisplayImage(element, src) {
  if (typeof src === "string" && src.length > 0) {
    element.src = src;
    element.classList.remove("is-empty");
    return;
  }
  element.removeAttribute("src");
  element.classList.add("is-empty");
}

function refreshOverlayStageState() {
  const hasImage =
    !elements.imgOriginal.classList.contains("is-empty") || !elements.imgOverlay.classList.contains("is-empty");
  elements.overlayStage.classList.toggle("has-image", hasImage);
}

function clearImages() {
  setDisplayImage(elements.imgOriginal, "");
  setDisplayImage(elements.imgOverlay, "");
  setDisplayImage(elements.imgHeatmap, "");
  setDisplayImage(elements.imgMask, "");
  elements.decisionMargin.textContent = "—";
  elements.confidenceBand.textContent = "—";
  elements.hotspotArea.textContent = "—";
  elements.warningText.textContent = "No warnings.";
  elements.summaryLabel.textContent = "—";
  elements.summaryProbability.textContent = "—";
  elements.summaryThreshold.textContent = "—";
  elements.summaryConfidenceBand.textContent = "—";
  elements.summaryBanner.className = "summary-banner neutral";
  elements.summaryBanner.textContent = "Awaiting prediction.";
  renderLimits();
  refreshOverlayStageState();
}

function buildClinicalBanner({
  label,
  confidenceBand,
  margin,
  malignantProbability,
  threshold,
  hotspotAreaPct,
  sourceKind,
  warnings,
}) {
  const normalizedLabel = String(label || "").toLowerCase();
  const normalizedBand = String(confidenceBand || "").toLowerCase();
  const normalizedWarnings = Array.isArray(warnings) ? warnings : [];
  const severeWarning = normalizedWarnings.some((w) =>
    /low-interpretability|very small activation|indeterminate|unstable/i.test(String(w)),
  );
  const softWarning = normalizedWarnings.length > 0 && !severeWarning;
  const marginAbs = typeof margin === "number" ? Math.abs(margin) : null;
  const nearThreshold = marginAbs !== null && marginAbs < 0.015;
  const lowConfidence = normalizedBand === "low" || (normalizedBand !== "high" && nearThreshold);
  const mediumConfidence = normalizedBand === "medium" || (marginAbs !== null && marginAbs < 0.06);
  const hasInterpretabilityRisk =
    severeWarning || (sourceKind === "dicom" && typeof hotspotAreaPct === "number" && hotspotAreaPct < 0.25);
  const benignStable =
    normalizedLabel === "benign" &&
    typeof malignantProbability === "number" &&
    typeof threshold === "number" &&
    malignantProbability <= Math.max(0.15, threshold * 1.8) &&
    normalizedBand === "high" &&
    !hasInterpretabilityRisk;

  if (normalizedLabel === "malignant" && !lowConfidence) {
    return {
      kind: "alert",
      text:
        "Elevated malignancy signal. Use High Recall triage pathway and prioritize expert review of this case.",
    };
  }
  if (benignStable) {
    return {
      kind: "good",
      text:
        "Stable low-risk signal in current mode. Continue routine review and confirm with radiologist interpretation.",
    };
  }
  if (softWarning) {
    return {
      kind: "neutral",
      text:
        "Model warning detected with otherwise stable output. Verify context and consider additional review before final triage.",
    };
  }
  if (hasInterpretabilityRisk) {
    return {
      kind: "caution",
      text:
        "Low-certainty or unstable pattern detected. Treat as indeterminate and require additional imaging review.",
    };
  }
  if (lowConfidence) {
    return {
      kind: "neutral",
      text:
        "Borderline decision around operating threshold. Confirm with additional views and clinical context before triage.",
    };
  }
  if (mediumConfidence) {
    return {
      kind: "neutral",
      text:
        "Intermediate certainty. Decision should be confirmed with clinical context, prior exams, and reader review.",
    };
  }
  return {
    kind: "good",
    text:
      "Consistent low-risk pattern in current mode. Keep standard workflow checks and confirm with radiologist judgment.",
  };
}

function updateSampleSelectionUI() {
  const cards = elements.sampleGrid.querySelectorAll(".sample-card");
  cards.forEach((card) => {
    card.classList.toggle("active", card.dataset.sampleId === state.selectedSampleId);
  });
}

function selectSample(sampleId) {
  state.selectedSampleId = sampleId;
  state.file = null;
  elements.fileInput.value = "";
  elements.fileName.textContent = "No file selected";
  updateSampleSelectionUI();
  setStatus("idle", `Sample selected: ${sampleId}`);
  setBusy(false);
}

function buildSampleCard(sample) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "sample-card";
  button.dataset.sampleId = sample.sample_id;
  button.innerHTML = `
    <img src="${sample.thumbnail_png}" alt="${sample.sample_id}" />
    <div class="sample-card-meta">
      <span class="sample-id">${sample.sample_id}</span>
      <span class="sample-chip ${sample.ground_truth === "malignant" ? "malignant" : "benign"}">${sample.ground_truth}</span>
    </div>
  `;
  button.addEventListener("click", () => selectSample(sample.sample_id));
  return button;
}

async function fetchSamples() {
  try {
    const resp = await fetch("/api/samples");
    if (!resp.ok) {
      elements.samplesMeta.textContent = "Unavailable";
      return;
    }
    const payload = await resp.json();
    const items = Array.isArray(payload.items) ? payload.items : [];
    state.samples = items;
    elements.sampleGrid.innerHTML = "";
    items.forEach((sample) => {
      elements.sampleGrid.appendChild(buildSampleCard(sample));
    });
    elements.samplesMeta.textContent = `${items.length} ready`;
    if (items.length > 0) {
      selectSample(items[0].sample_id);
    } else {
      setStatus("idle", "Upload file to start. Demo gallery unavailable.");
      setBusy(false);
    }
  } catch (_err) {
    elements.samplesMeta.textContent = "Unavailable";
    setStatus("error", "Could not load demo gallery.");
  }
}

function wireFileInput() {
  elements.fileInput.addEventListener("change", () => {
    const [file] = elements.fileInput.files || [];
    if (!file) return;
    state.file = file;
    state.selectedSampleId = null;
    updateSampleSelectionUI();
    elements.fileName.textContent = `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    setStatus("idle", "File ready");
    setBusy(false);
  });

  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.style.borderColor = "rgba(99, 243, 217, 1)";
  });
  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.style.borderColor = "";
  });
  elements.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropzone.style.borderColor = "";
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    elements.fileInput.files = event.dataTransfer.files;
    state.file = file;
    state.selectedSampleId = null;
    updateSampleSelectionUI();
    elements.fileName.textContent = `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    setStatus("idle", "File dropped");
    setBusy(false);
  });
}

async function fetchModes() {
  try {
    const resp = await fetch("/api/model/info");
    if (!resp.ok) return;
    const data = await resp.json();
    if (Array.isArray(data.available_modes) && data.available_modes.length > 0) {
      state.availableModes = data.available_modes;
      state.modeProfiles = typeof data.mode_profiles === "object" && data.mode_profiles ? data.mode_profiles : {};
      if (!state.availableModes.includes(state.mode)) {
        state.mode = typeof data.default_mode === "string" ? data.default_mode : state.availableModes[0];
      }
      activateMode(state.mode);
    }
  } catch (_err) {
    setStatus("error", "Model info endpoint is unavailable.");
  }
}

async function runPrediction() {
  if (!hasInput() || state.busy) return;
  setBusy(true);
  setStatus("loading", "Running inference...");

  try {
    let resp;
    if (state.selectedSampleId) {
      resp = await fetch(`/api/predict/sample/${encodeURIComponent(state.selectedSampleId)}?mode=${encodeURIComponent(state.mode)}`, {
        method: "POST",
      });
    } else {
      const form = new FormData();
      form.append("file", state.file);
      form.append("mode", state.mode);
      resp = await fetch("/api/predict", {
        method: "POST",
        body: form,
      });
    }

    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.detail || "Prediction failed.");

    const pred = payload.prediction;
    const explain = payload.explainability?.images || {};
    const explainStats = payload.explainability?.stats || {};
    const model = payload.model || {};
    const decision = payload.decision || {};
    const warnings = Array.isArray(payload.warnings) ? payload.warnings : [];

    elements.predictionLabel.textContent = pred?.label || "—";
    elements.confidenceLabel.textContent = typeof pred?.confidence === "number" ? formatPct(pred.confidence) : "—";
    const thresholdText = typeof model.threshold === "number" ? model.threshold.toFixed(3) : "—";
    const gtText = payload.sample?.ground_truth ? ` • gt: ${payload.sample.ground_truth}` : "";
    const pmText = typeof pred?.malignant_probability === "number" ? formatPct(pred.malignant_probability) : "—";
    const latencyText = typeof payload.timing_ms === "number" ? `${payload.timing_ms.toFixed(0)}ms` : "—ms";
    elements.resultMeta.textContent =
      `p(malignant): ${pmText} • threshold: ${thresholdText} • ${latencyText}${gtText}`;
    const margin = typeof decision.delta_to_threshold === "number" ? decision.delta_to_threshold : null;
    elements.decisionMargin.textContent = margin === null ? "—" : `${margin >= 0 ? "+" : ""}${margin.toFixed(3)}`;
    elements.confidenceBand.textContent = decision.confidence_band || "—";
    elements.summaryLabel.textContent = pred?.label ? pred.label.toUpperCase() : "—";
    elements.summaryProbability.textContent = pmText;
    elements.summaryThreshold.textContent = thresholdText;
    elements.summaryConfidenceBand.textContent = decision.confidence_band ? String(decision.confidence_band).toUpperCase() : "—";
    elements.hotspotArea.textContent =
      typeof explainStats.hotspot_area_pct === "number" ? `${explainStats.hotspot_area_pct.toFixed(2)}%` : "—";
    elements.warningText.textContent =
      warnings.length > 0 ? warnings.map((w, i) => `${i + 1}) ${w}`).join("  ") : "No warnings.";
    renderLimits(warnings.slice(0, 2));

    const banner = buildClinicalBanner({
      label: pred?.label || "",
      confidenceBand: decision.confidence_band || "",
      margin,
      malignantProbability: typeof pred?.malignant_probability === "number" ? pred.malignant_probability : null,
      threshold: typeof model.threshold === "number" ? model.threshold : null,
      hotspotAreaPct: Number(explainStats.hotspot_area_pct || 0),
      sourceKind: payload.source_kind || "",
      warnings,
    });
    elements.summaryBanner.className = `summary-banner ${banner.kind}`;
    elements.summaryBanner.textContent = banner.text;

    setDisplayImage(elements.imgOriginal, explain.original || "");
    setDisplayImage(elements.imgOverlay, explain.overlay || "");
    setDisplayImage(elements.imgHeatmap, explain.heatmap || "");
    setDisplayImage(elements.imgMask, explain.mask || "");
    refreshOverlayStageState();

    const confidenceText = typeof pred?.confidence === "number" ? formatPct(pred.confidence) : "—";
    const labelText = pred?.label ? pred.label.toUpperCase() : "UNKNOWN";
    setStatus("ok", `Done • ${labelText} (${confidenceText})`);
  } catch (err) {
    setStatus("error", err instanceof Error ? err.message : "Unknown error");
  } finally {
    setBusy(false);
  }
}

function wireControls() {
  elements.modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => activateMode(btn.dataset.mode));
  });
  elements.predictButton.addEventListener("click", runPrediction);
  elements.overlayAlpha.addEventListener("input", () => {
    const alpha = Number(elements.overlayAlpha.value) / 100;
    elements.imgOverlay.style.opacity = String(alpha);
    refreshOverlayStageState();
  });
}

function initThreeBackground() {
  const canvas = document.getElementById("bg-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x050a15, 0.065);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 9);

  const ambient = new THREE.AmbientLight(0x4ba4ff, 0.38);
  const key = new THREE.PointLight(0x63f3d9, 1.9, 24, 2);
  key.position.set(2.6, 2.1, 3.2);
  const rim = new THREE.PointLight(0xff8b6a, 1.2, 22, 2);
  rim.position.set(-2.8, -1.2, 1.8);
  scene.add(ambient, key, rim);

  const coreGroup = new THREE.Group();
  scene.add(coreGroup);

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.05, 1),
    new THREE.MeshPhysicalMaterial({
      color: 0x56dcff,
      emissive: 0x08273c,
      roughness: 0.28,
      metalness: 0.65,
      clearcoat: 1,
      transparent: true,
      opacity: 0.72,
      wireframe: true,
    }),
  );
  coreGroup.add(core);

  const ringA = new THREE.Mesh(
    new THREE.TorusGeometry(1.85, 0.04, 14, 180),
    new THREE.MeshBasicMaterial({ color: 0x63f3d9, transparent: true, opacity: 0.42 }),
  );
  ringA.rotation.x = 1.24;
  ringA.rotation.z = 0.32;
  coreGroup.add(ringA);

  const ringB = new THREE.Mesh(
    new THREE.TorusGeometry(2.32, 0.055, 16, 200),
    new THREE.MeshBasicMaterial({ color: 0x45a3ff, transparent: true, opacity: 0.28 }),
  );
  ringB.rotation.y = 1.1;
  ringB.rotation.x = 0.34;
  coreGroup.add(ringB);

  const helixGroup = new THREE.Group();
  scene.add(helixGroup);

  const pointsA = [];
  const pointsB = [];
  for (let i = 0; i < 240; i += 1) {
    const t = i * 0.19;
    const y = (i - 120) * 0.03;
    pointsA.push(new THREE.Vector3(Math.cos(t) * 2.15, y, Math.sin(t) * 2.15));
    pointsB.push(new THREE.Vector3(Math.cos(t + Math.PI) * 2.15, y, Math.sin(t + Math.PI) * 2.15));
  }

  const helixA = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pointsA),
    new THREE.LineBasicMaterial({ color: 0x63f3d9, transparent: true, opacity: 0.56 }),
  );
  const helixB = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pointsB),
    new THREE.LineBasicMaterial({ color: 0xff8b6a, transparent: true, opacity: 0.48 }),
  );
  helixGroup.add(helixA, helixB);

  const bridgePositions = [];
  for (let i = 0; i < pointsA.length; i += 8) {
    bridgePositions.push(pointsA[i].x, pointsA[i].y, pointsA[i].z);
    bridgePositions.push(pointsB[i].x, pointsB[i].y, pointsB[i].z);
  }
  const bridgeGeo = new THREE.BufferGeometry();
  bridgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(bridgePositions, 3));
  const bridges = new THREE.LineSegments(
    bridgeGeo,
    new THREE.LineBasicMaterial({ color: 0x7fb9ff, transparent: true, opacity: 0.3 }),
  );
  helixGroup.add(bridges);

  const nebulaGroup = new THREE.Group();
  scene.add(nebulaGroup);

  function makeNebula(size, color, x, y, z, opacity) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    mesh.position.set(x, y, z);
    return mesh;
  }

  const nebulaA = makeNebula(9.2, 0x144080, -3.4, 1.8, -5.5, 0.12);
  const nebulaB = makeNebula(7.6, 0x0f7e70, 3.8, -2.1, -4.8, 0.1);
  const nebulaC = makeNebula(6.8, 0x7a2f7f, 0.2, 2.5, -6.2, 0.08);
  nebulaGroup.add(nebulaA, nebulaB, nebulaC);

  const starCount = 3600;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i += 1) {
    const i3 = i * 3;
    starPositions[i3] = (Math.random() - 0.5) * 25;
    starPositions[i3 + 1] = (Math.random() - 0.5) * 18;
    starPositions[i3 + 2] = (Math.random() - 0.5) * 22;
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  const stars = new THREE.Points(
    starGeo,
    new THREE.PointsMaterial({ color: 0x92d1ff, size: 0.02, transparent: true, opacity: 0.72 }),
  );
  scene.add(stars);

  const parallax = { x: 0, y: 0, tx: 0, ty: 0 };
  window.addEventListener("pointermove", (event) => {
    const nx = event.clientX / window.innerWidth - 0.5;
    const ny = event.clientY / window.innerHeight - 0.5;
    parallax.tx = nx;
    parallax.ty = ny;
  });

  const clock = new THREE.Clock();
  function render() {
    const t = clock.getElapsedTime();

    parallax.x += (parallax.tx - parallax.x) * 0.035;
    parallax.y += (parallax.ty - parallax.y) * 0.035;

    coreGroup.rotation.x = t * 0.18;
    coreGroup.rotation.y = t * 0.23;
    helixGroup.rotation.y = -t * 0.12;
    helixGroup.rotation.z = Math.sin(t * 0.22) * 0.08;

    nebulaA.rotation.z = t * 0.014;
    nebulaB.rotation.z = -t * 0.012;
    nebulaC.rotation.z = t * 0.016;

    stars.rotation.y = -t * 0.011;
    stars.rotation.x = Math.sin(t * 0.12) * 0.06;

    camera.position.x = parallax.x * 0.9;
    camera.position.y = -parallax.y * 0.55;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  window.addEventListener("resize", onResize);
  onResize();
  render();
}

async function bootstrap() {
  initThreeBackground();
  clearImages();
  wireFileInput();
  wireControls();
  await fetchSamples();
  await fetchModes();
  activateMode(state.mode);
  setBusy(false);
}

bootstrap();
