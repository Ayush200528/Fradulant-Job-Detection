"""
Flask backend for the Job Posting Fraud Checker.

Serves:
  GET  /                -> the frontend (index.html)
  POST /predict          -> { verdict, probability, top_features, red_flags, extracted_preview? }

Run locally:
    pip install flask joblib xgboost shap scikit-learn requests beautifulsoup4
    python backend.py
Then open http://localhost:5000 in your browser.
"""

import os
import re
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- Resolve model paths relative to THIS script's location ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")  # adjust if your layout differs

# ---------- Load pipeline once at startup ----------
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))
print("Model pipeline loaded.")

RED_FLAG_PHRASES = [
    "wire transfer", "processing fee", "no experience needed",
    "send your bank", "western union", "money gram", "registration fee",
    "starter kit", "upfront payment", "click this link", "urgent hiring",
    "work from home no interview",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_red_flags(raw_text: str):
    lower = raw_text.lower()
    return [p for p in RED_FLAG_PHRASES if p in lower]


def fetch_posting_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    mode = data.get("mode", "text")
    threshold = float(data.get("threshold", 0.25))
    extracted_preview = None

    if mode == "url":
        url = data.get("content", "").strip()
        if not url:
            return jsonify({"error": "No URL provided."}), 400
        try:
            raw_text = fetch_posting_text(url)
        except Exception as e:
            return jsonify({"error": f"Couldn't fetch that URL ({e}). The site may "
                                      f"block automated requests, require login, or "
                                      f"the link may be incorrect — try pasting text "
                                      f"instead."}), 400
        extracted_preview = raw_text[:1500] + ("..." if len(raw_text) > 1500 else "")
    else:
        raw_text = data.get("content", "").strip()
        if not raw_text:
            return jsonify({"error": "No text provided."}), 400

    cleaned = clean_text(raw_text)
    vec = tfidf.transform([cleaned])

    prob_fraud = float(model.predict_proba(vec)[0][1])
    is_fraud = prob_fraud >= threshold

    # ---------- SHAP explanation ----------
    shap_values = explainer(vec)
    feature_names = np.array(tfidf.get_feature_names_out())
    contributions = shap_values.values[0]
    nonzero_idx = vec.nonzero()[1]
    pairs = [(feature_names[i], float(contributions[i])) for i in nonzero_idx]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [
        {"word": w, "impact": round(v, 3), "direction": "fraud" if v > 0 else "legit"}
        for w, v in pairs[:8]
    ]

    return jsonify({
        "verdict": "fraudulent" if is_fraud else "legitimate",
        "probability": round(prob_fraud, 4),
        "threshold": threshold,
        "top_features": top_features,
        "red_flags": check_red_flags(raw_text),
        "extracted_preview": extracted_preview,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)