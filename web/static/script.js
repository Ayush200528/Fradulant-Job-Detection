// ---------- Element refs ----------
const tabs = document.querySelectorAll(".tab");
const tabIndicator = document.getElementById("tab-indicator");
const textInput = document.getElementById("text-input");
const urlWrap = document.getElementById("url-input-wrap");
const urlInput = document.getElementById("url-input");
const thresholdSlider = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");
const resultSection = document.getElementById("result-section");
const urlPreview = document.getElementById("url-preview");
const urlPreviewText = document.getElementById("url-preview-text");
const scanOverlay = document.getElementById("scan-overlay");
const caseNumberEl = document.getElementById("case-number");
const sessionCountEl = document.getElementById("session-count");
const verdictCard = document.getElementById("verdict-card");
const verdictBadge = document.getElementById("verdict-badge");
const gaugeFill = document.getElementById("gauge-fill");

let mode = "text";
let caseCounter = Math.floor(1000 + Math.random() * 8999);
let sessionCount = 0;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Gauge geometry (r = 84 in the SVG)
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 84;
gaugeFill.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE}`;
gaugeFill.style.strokeDashoffset = `${GAUGE_CIRCUMFERENCE}`;

// ---------- Case reference ----------
function nextCaseId() {
  caseCounter += 1;
  const stamp = Date.now().toString(36).slice(-3).toUpperCase();
  return `${caseCounter}-${stamp}`;
}
caseNumberEl.textContent = `REF ${nextCaseId()}`;

// ---------- Tab switching with sliding indicator ----------
function positionIndicator(tabEl) {
  tabIndicator.style.width = tabEl.offsetWidth + "px";
  tabIndicator.style.transform = `translateX(${tabEl.offsetLeft}px)`;
}
window.addEventListener("load", () => positionIndicator(document.querySelector(".tab.active")));

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    positionIndicator(tab);
    mode = tab.dataset.mode;
    if (mode === "text") {
      textInput.classList.remove("hidden");
      urlWrap.classList.add("hidden");
    } else {
      textInput.classList.add("hidden");
      urlWrap.classList.remove("hidden");
    }
    hideError();
  });
});

// ---------- Threshold slider with filled track ----------
function updateSliderFill() {
  const min = parseFloat(thresholdSlider.min);
  const max = parseFloat(thresholdSlider.max);
  const val = parseFloat(thresholdSlider.value);
  const pct = ((val - min) / (max - min)) * 100;
  thresholdSlider.style.background =
    `linear-gradient(90deg, var(--accent-2) 0%, var(--accent) ${pct}%, var(--track) ${pct}%, var(--track) 100%)`;
}
thresholdSlider.addEventListener("input", () => {
  thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
  updateSliderFill();
});
updateSliderFill();

function hideError() {
  errorMsg.classList.add("hidden");
  errorMsg.textContent = "";
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
}

// ---------- Submit ----------
submitBtn.addEventListener("click", async () => {
  hideError();
  const content = mode === "text" ? textInput.value.trim() : urlInput.value.trim();
  if (!content) {
    showError(mode === "text" ? "Paste some job posting text first." : "Enter a URL first.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.classList.add("loading");
  scanOverlay.classList.add("active");
  resultSection.classList.add("hidden");
  resultSection.classList.remove("showing");
  urlPreview.classList.add("hidden");

  const minSpinner = new Promise((resolve) => setTimeout(resolve, 550));

  try {
    const [res] = await Promise.all([
      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          content,
          threshold: parseFloat(thresholdSlider.value),
        }),
      }),
      minSpinner,
    ]);
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }

    renderResult(data);
    sessionCount += 1;
    sessionCountEl.textContent = sessionCount;
  } catch (err) {
    showError("Couldn't reach the server. Is backend.py running?");
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("loading");
    scanOverlay.classList.remove("active");
  }
});

// ---------- Count-up animation for confidence % ----------
function animateCount(el, target, duration = 800) {
  if (reducedMotion) {
    el.textContent = target.toFixed(1) + "%";
    return;
  }
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (target * eased).toFixed(1) + "%";
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = target.toFixed(1) + "%";
  }
  requestAnimationFrame(tick);
}

// ---------- Gauge fill animation ----------
function setGauge(pct, isFraud) {
  const clamped = Math.max(0, Math.min(100, pct));
  const offset = GAUGE_CIRCUMFERENCE * (1 - clamped / 100);
  gaugeFill.classList.toggle("fraud", isFraud);
  if (reducedMotion) {
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDashoffset = `${offset}`;
    return;
  }
  // reset then animate on next frame
  gaugeFill.style.transition = "none";
  gaugeFill.style.strokeDashoffset = `${GAUGE_CIRCUMFERENCE}`;
  requestAnimationFrame(() => {
    gaugeFill.style.transition = "stroke-dashoffset 900ms cubic-bezier(0.22,1,0.36,1)";
    gaugeFill.style.strokeDashoffset = `${offset}`;
  });
}

// ---------- Render ----------
function renderResult(data) {
  const isFraud = data.verdict === "fraudulent";
  const confidencePct = isFraud ? data.probability * 100 : (1 - data.probability) * 100;

  verdictCard.classList.remove("legit", "fraud");
  verdictCard.classList.add(isFraud ? "fraud" : "legit");

  verdictBadge.className = "verdict-badge " + (isFraud ? "fraud" : "legit");
  verdictBadge.textContent = isFraud ? "Potentially fraudulent" : "Likely legitimate";

  setGauge(confidencePct, isFraud);

  const confidenceEl = document.getElementById("confidence-value");
  confidenceEl.textContent = "0%";
  animateCount(confidenceEl, confidencePct);

  document.getElementById("verdict-detail").textContent = isFraud
    ? "This posting shows patterns consistent with fraudulent listings."
    : "This posting looks consistent with legitimate listings.";
  document.getElementById("verdict-case").textContent =
    `${caseNumberEl.textContent} · threshold ${data.threshold.toFixed(2)}`;

  // SHAP-style feature bars, scaled to the largest absolute impact
  const featureList = document.getElementById("feature-list");
  featureList.innerHTML = "";
  const maxImpact = data.top_features.reduce(
    (m, f) => Math.max(m, Math.abs(f.impact)), 0) || 1;

  data.top_features.forEach((f, i) => {
    const li = document.createElement("li");
    li.className = "feature-item " + f.direction;
    li.style.animationDelay = `${i * 45}ms`;
    const width = Math.max(6, (Math.abs(f.impact) / maxImpact) * 100);
    li.innerHTML = `
      <span class="feature-word">${escapeHtml(f.word)}</span>
      <span class="feature-bar-wrap">
        <span class="feature-bar" style="width:${width}%"></span>
      </span>
      <span class="feature-meta">
        <span class="feature-dir">${f.direction === "fraud" ? "fraud" : "legit"}</span>
        <span class="feature-impact mono">${f.impact > 0 ? "+" : ""}${f.impact}</span>
      </span>
    `;
    featureList.appendChild(li);
  });

  const redflagSection = document.getElementById("redflag-section");
  const redflagList = document.getElementById("redflag-list");
  redflagList.innerHTML = "";
  if (data.red_flags && data.red_flags.length > 0) {
    data.red_flags.forEach((phrase, i) => {
      const li = document.createElement("li");
      li.textContent = phrase;
      li.style.animationDelay = `${data.top_features.length * 45 + i * 45}ms`;
      redflagList.appendChild(li);
    });
    redflagSection.classList.remove("hidden");
  } else {
    redflagSection.classList.add("hidden");
  }

  if (data.extracted_preview) {
    urlPreviewText.textContent = data.extracted_preview;
    urlPreview.classList.remove("hidden");
  }

  resultSection.classList.remove("hidden");
  requestAnimationFrame(() => resultSection.classList.add("showing"));
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}