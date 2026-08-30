"""
Train the fraud job detector with TEXT (TF-IDF) + EXPLICIT RED-FLAG FEATURES,
combined -> SMOTE -> XGBoost -> SHAP.

WHY THIS CHANGED FROM THE FIRST VERSION:
The original model relied only on TF-IDF word patterns. On postings unlike
anything in the training data (e.g. a detailed international finance role),
it had no strong signal to use and fell back on generic filler words
("would", "of", "we"), producing a low-confidence, low-quality "fraud" call.
Adding explicit, hand-picked red-flag features gives the model something
concrete to anchor on regardless of overall writing style.

Run locally:
    pip install xgboost shap imbalanced-learn scikit-learn pandas joblib scipy
    python train_model.py
"""

import re
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# ---------- Red-flag phrases: MUST match backend.py exactly ----------
# (kept in one place here; backend.py has its own copy that must stay in sync)
RED_FLAG_PHRASES = [
    "wire transfer", "processing fee", "no experience needed",
    "send your bank", "western union", "money gram", "registration fee",
    "starter kit", "upfront payment", "click this link", "urgent hiring",
    "work from home no interview", "guaranteed income", "easy money",
    "personal email", "whatsapp", "telegram", "social security number",
    "credit card", "gift card",
]

def red_flag_features(text: str) -> list:
    """Returns a 0/1 vector, one entry per phrase in RED_FLAG_PHRASES."""
    lower = text.lower()
    return [1 if phrase in lower else 0 for phrase in RED_FLAG_PHRASES]

# ---------- 1. Load cleaned data ----------
df = pd.read_csv("processed_data/preprocessed_cleaned.csv")
X_text = df["combined_text"]
y = df["fraudulent"]

# ---------- 2. Split first (unchanged — prevents leakage) ----------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 3. TF-IDF (fit on train only, unchanged) ----------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# ---------- 4. NEW: compute red-flag binary features for every posting ----------
X_train_flags = np.array([red_flag_features(t) for t in X_train_text])
X_test_flags = np.array([red_flag_features(t) for t in X_test_text])
print(f"Red-flag features: {len(RED_FLAG_PHRASES)} columns added")

# ---------- 5. NEW: combine TF-IDF (sparse) + red-flag features (dense->sparse) ----------
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_flags)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_flags)])

# ---------- 6. SMOTE on the combined training features only ----------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_combined, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_bal).value_counts().to_dict()}")

# ---------- 7. XGBoost (unchanged params) ----------
clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    eval_metric="logloss", random_state=42,
)
clf.fit(X_train_bal, y_train_bal)

# ---------- 8. Evaluate on the untouched, imbalanced test set ----------
preds = clf.predict(X_test_combined)
probs = clf.predict_proba(X_test_combined)[:, 1]
print("\n=== Test set performance (with red-flag features) ===")
print(classification_report(y_test, preds, digits=3))
print(f"PR-AUC: {average_precision_score(y_test, probs):.3f}")

# ---------- 9. SHAP explainer on the COMBINED feature space ----------
explainer = shap.TreeExplainer(clf)

# ---------- 10. Save everything, including the combined feature name list ----------
# This ordering (tfidf vocab, then red-flag names) MUST match what backend.py builds.
feature_names = list(tfidf.get_feature_names_out()) + RED_FLAG_PHRASES

joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(clf, "xgb_model.joblib")
joblib.dump(explainer, "shap_explainer.joblib")
joblib.dump(feature_names, "feature_names.joblib")
print("\nSaved: tfidf_vectorizer.joblib, xgb_model.joblib, shap_explainer.joblib, feature_names.joblib")