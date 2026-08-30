"""
Generate a confusion matrix for the current trained model, at whatever
threshold you're using in the app (default 0.25).

Run locally:
    pip install matplotlib seaborn
    python confusion_matrix.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report

# ---------- Paths resolved relative to THIS script's location ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")          # adjust if needed
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "processed_data", "preprocessed_cleaned.csv")  # adjust if needed

# ---------- Threshold to evaluate at (match your app's current setting) ----------
THRESHOLD = 0.25

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
preds = (probs >= THRESHOLD).astype(int)

# ---------- Confusion matrix ----------
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion matrix at threshold {THRESHOLD}:\n")
print(f"                 Predicted Legit   Predicted Fraud")
print(f"Actual Legit          {tn:>6}            {fp:>6}")
print(f"Actual Fraud          {fn:>6}            {tp:>6}")

print(f"\nTrue Negatives  (correctly caught legit):  {tn}")
print(f"False Positives (legit flagged as fraud):  {fp}")
print(f"False Negatives (fraud missed entirely):   {fn}")
print(f"True Positives  (correctly caught fraud):  {tp}")

print(f"\nOf {tn + fp} real legitimate postings, {fp} were wrongly flagged ({fp/(tn+fp)*100:.1f}%).")
print(f"Of {fn + tp} real fraudulent postings, {fn} were missed entirely ({fn/(fn+tp)*100:.1f}%).")

# ---------- Precision, Recall, F1 (fraud class) ----------
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

print(f"\n=== Metrics at threshold {THRESHOLD} (fraud class) ===")
print(f"Precision: {precision:.3f}  (of postings flagged fraud, how many actually were)")
print(f"Recall:    {recall:.3f}  (of actual fraud postings, how many were caught)")
print(f"F1 score:  {f1:.3f}  (harmonic balance of precision and recall)")

print("\n=== Full classification report ===")
print(classification_report(y_test, preds, target_names=["Legitimate", "Fraudulent"], digits=3))

# ---------- Plot and save ----------
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraudulent"])
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix (threshold = {THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "confusion_matrix.png"), dpi=150)
print(f"\nSaved plot to confusion_matrix.png")
plt.show()