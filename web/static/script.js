// ---------- Splash ----------
const splash = document.getElementById("splash");
const splashBtn = document.getElementById("splash-continue");
function dismissSplash() {
  splash.classList.add("dismissed");
  document.body.classList.remove("splash-locked");
}
splashBtn.addEventListener("click", dismissSplash);

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
const verdictStage = document.getElementById("verdict-stage");
const confettiLayer = document.getElementById("confetti-layer");

let mode = "text";
let caseCounter = Math.floor(1000 + Math.random() * 8999);
let sessionCount = 0;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const CONFETTI_COLORS = ["#ffb703", "#ff6b9d", "#7c5cff", "#1fe0a8", "#ffffff"];

function nextCaseId() {
  caseCounter += 1;
  const stamp = Date.now().toString(36).slice(-3).toUpperCase();
  return `${caseCounter}-${stamp}`;
}
caseNumberEl.textContent = `CASE NO. ${nextCaseId()}`;

// ---------- Cycling headline word ----------
const CYCLE_WORDS = ["job posting", "recruiter message", "offer email", "DM offer"];
let cycleIndex = 0;
const cycleEl = document.getElementById("cycle-word");
if (!reducedMotion) {
  setInterval(() => {
    cycleEl.classList.add("swap-out");
    setTimeout(() => {
      cycleIndex = (cycleIndex + 1) % CYCLE_WORDS.length;
      cycleEl.textContent = CYCLE_WORDS[cycleIndex];
      cycleEl.classList.remove("swap-out");
    }, 350);
  }, 2400);
}

// ---------- Tab switching with sliding indicator ----------
function positionIndicator(tabEl) {
  tabIndicator.style.width = tabEl.offsetWidth + "px";
  tabIndicator.style.transform = `translateX(${tabEl.offsetLeft}px)`;
}
window.addEventListener("load", () => positionIndicator(document.querySelector(".tab.active")));

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
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
    `linear-gradient(90deg, var(--accent-2) 0%, var(--accent) ${pct}%, var(--border) ${pct}%, var(--border) 100%)`;
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
function animateCount(el, target, duration = 700) {
  if (reducedMotion) {
    el.textContent = target.toFixed(1) + "%";
    return;
  }
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = (target * eased).toFixed(1);
    el.textContent = value + "%";
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = target.toFixed(1) + "%";
  }
  requestAnimationFrame(tick);
}

// ---------- Confetti burst (legit verdict) ----------
function fireConfetti() {
  if (reducedMotion) return;
  confettiLayer.innerHTML = "";
  const originX = 64; // roughly center of the avatar column
  for (let i = 0; i < 24; i++) {
    const piece = document.createElement("div");
    piece.className = "confetti-piece";
    const color = CONFETTI_COLORS[Math.floor(Math.random() * CONFETTI_COLORS.length)];
    piece.style.background = color;
    piece.style.left = (originX + (Math.random() * 100 - 50)) + "px";
    piece.style.animationDelay = (Math.random() * 0.2) + "s";
    piece.style.animationDuration = (0.9 + Math.random() * 0.6) + "s";
    piece.style.borderRadius = Math.random() > 0.5 ? "50%" : "2px";
    confettiLayer.appendChild(piece);
  }
  setTimeout(() => { confettiLayer.innerHTML = ""; }, 1800);
}

// ---------- Render ----------
function renderResult(data) {
  const isFraud = data.verdict === "fraudulent";
  const confidencePct = isFraud ? data.probability * 100 : (1 - data.probability) * 100;

  const avatarWrap = document.getElementById("avatar-wrap");
  avatarWrap.classList.remove("landing", "legit", "fraud");
  void avatarWrap.offsetWidth; // restart animation

  verdictStage.classList.toggle("fraud-glow", isFraud);
  verdictStage.classList.remove("shake-fraud");

  if (isFraud) {
    avatarWrap.classList.add("fraud");
    void verdictStage.offsetWidth;
    verdictStage.classList.add("shake-fraud");
  } else {
    avatarWrap.classList.add("legit");
  }
  requestAnimationFrame(() => avatarWrap.classList.add("landing"));

  if (!isFraud) {
    setTimeout(fireConfetti, 150);
  }

  const confidenceEl = document.getElementById("confidence-value");
  confidenceEl.textContent = "0%";
  animateCount(confidenceEl, confidencePct);

  document.getElementById("verdict-detail").textContent = isFraud
    ? "Likely fraudulent posting"
    : "Likely legitimate posting";
  document.getElementById("verdict-case").textContent =
    `${caseNumberEl.textContent} — threshold ${data.threshold.toFixed(2)}`;

  const featureList = document.getElementById("feature-list");
  featureList.innerHTML = "";
  data.top_features.forEach((f, i) => {
    const li = document.createElement("li");
    li.className = "feature-item";
    li.style.animationDelay = `${i * 45}ms`;
    li.innerHTML = `
      <span class="feature-word">${escapeHtml(f.word)}</span>
      <span style="display:flex; align-items:center;">
        <span class="feature-dir ${f.direction}">${f.direction === "fraud" ? "→ fraud" : "→ legit"}</span>
        <span class="feature-impact">${f.impact > 0 ? "+" : ""}${f.impact}</span>
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
      li.textContent = `"${phrase}"`;
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