"""
Sweep classification thresholds on the held-out test set to find the best
cutoff for YOUR priority (recall vs precision), instead of defaulting to 0.5.

UPDATED: now builds the same TF-IDF + red-flag combined feature vector that
train_model.py trains on and backend.py predicts on. Running this against
the old TF-IDF-only vector will fail with a feature-shape mismatch, since
the retrained model expects 5020 columns (5000 TF-IDF + 20 red-flag flags).

Run this after train_model.py (reuses the same saved vectorizer + model).

Run locally:
    python threshold_sweep.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------- Paths resolved relative to THIS script's location ----------
# so it works no matter which folder you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "processed_data", "preprocessed_cleaned.csv")

# ---------- MUST exactly match train_model.py / backend.py, same order ----------
RED_FLAG_PHRASES = [
    "wire transfer", "processing fee", "no experience needed",
    "send your bank", "western union", "money gram", "registration fee",
    "starter kit", "upfront payment", "click this link", "urgent hiring",
    "work from home no interview", "guaranteed income", "easy money",
    "personal email", "whatsapp", "telegram", "social security number",
    "credit card", "gift card",
]

def red_flag_features(text: str) -> list:
    lower = text.lower()
    return [1 if phrase in lower else 0 for phrase in RED_FLAG_PHRASES]

# ---------- Load data the SAME way train_model.py did ----------
df = pd.read_csv(DATA_PATH)
X = df["combined_text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- Load saved pipeline ----------
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))

# ---------- Build the SAME combined feature vector the model was trained on ----------
X_test_tfidf = tfidf.transform(X_test)
X_test_flags = np.array([red_flag_features(t) for t in X_test])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_flags)])

probs = model.predict_proba(X_test_combined)[:, 1]
print("Probability predictions generated.")

# ---------- Sweep thresholds ----------
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 48)

results = []
for t in np.arange(0.1, 0.95, 0.05):
    preds = (probs >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    results.append((t, p, r, f1))
    print(f"{t:>10.2f} | {p:>10.3f} | {r:>10.3f} | {f1:>10.3f}")

# ---------- Highlight the best F1 threshold ----------
best = max(results, key=lambda x: x[3])
print(f"\nBest F1 threshold: {best[0]:.2f} "
      f"(precision={best[1]:.3f}, recall={best[2]:.3f}, f1={best[3]:.3f})")

# ---------- Also show a recall-prioritized option (>=80% recall) ----------
recall_priority = [r for r in results if r[2] >= 0.80]
if recall_priority:
    best_recall_option = max(recall_priority, key=lambda x: x[1])
    print(f"Best option with recall >= 0.80: threshold={best_recall_option[0]:.2f} "
          f"(precision={best_recall_option[1]:.3f}, recall={best_recall_option[2]:.3f})")
else:
    print("No threshold reaches 80% recall — consider retraining with more SMOTE "
          "neighbors or a lower decision threshold floor.")