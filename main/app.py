"""
Real-time fraudulent job posting detector.

Run locally:
    pip install streamlit shap xgboost scikit-learn joblib requests beautifulsoup4
"""

import re
import os
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

st.set_page_config(page_title="Job Posting Fraud Checker", page_icon="🔍")

# ---------- Resolve model paths relative to THIS script's location, ----------
# ---------- so it works no matter what folder you run streamlit from  ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")  # adjust if your folder layout differs

# ---------- Load saved pipeline (from train_model.py) ----------
@st.cache_resource
def load_pipeline():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    return tfidf, model, explainer

tfidf, model, explainer = load_pipeline()

# ---------- Same cleaning used during training (must match!) ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Simple heuristic red flags for a human-readable extra reason ----------
RED_FLAG_PHRASES = [
    "wire transfer", "processing fee", "no experience needed",
    "send your bank", "western union", "money gram", "registration fee",
    "starter kit", "upfront payment", "click this link", "urgent hiring",
    "work from home no interview",
]

def check_red_flags(raw_text: str):
    lower = raw_text.lower()
    return [p for p in RED_FLAG_PHRASES if p in lower]

# ---------- Fetch + extract visible text from a job posting URL ----------
def fetch_posting_text(url: str) -> str:
    headers = {
        # A normal browser user-agent; some sites block requests with no/blank UA.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip non-content tags before extracting text
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- UI ----------
st.title("🔍 Job Posting Fraud Checker")
st.write("Paste a job posting, or a link to one, and get an instant verdict with reasons.")

input_mode = st.radio("Input method", ["Paste text", "Paste URL"], horizontal=True)

job_text = ""

if input_mode == "Paste text":
    job_text = st.text_area("Job posting text", height=300, placeholder="Paste the full job posting here...")
else:
    url = st.text_input("Job posting URL", placeholder="https://company.com/careers/job-123")
    st.caption(
        "⚠️ Works for public company career pages and most job boards. "
        "LinkedIn actively blocks automated fetching, and links to private "
        "inboxes (Gmail/Outlook) can never be fetched this way — copy-paste "
        "those into 'Paste text' instead."
    )
    if url:
        with st.spinner("Fetching posting..."):
            try:
                job_text = fetch_posting_text(url)
                with st.expander("Preview extracted text"):
                    st.write(job_text[:1500] + ("..." if len(job_text) > 1500 else ""))
            except Exception as e:
                st.error(
                    f"Couldn't fetch that URL ({e}). "
                    "The site may block automated requests, require login, "
                    "or the link may be incorrect — try 'Paste text' instead."
                )

threshold = st.slider(
    "Fraud probability threshold", 0.1, 0.9, 0.25, 0.05,
    help="Default (0.25) is tuned from a precision/recall sweep on the test set — "
         "balances catching fraud (recall) against false alarms (precision). "
         "Lower = catches more fraud but more false alarms; higher = fewer false "
         "alarms but misses more fraud."
)

if st.button("Check this posting", type="primary") and job_text.strip():
    cleaned = clean_text(job_text)
    vec = tfidf.transform([cleaned])

    prob_fraud = model.predict_proba(vec)[0][1]
    is_fraud = prob_fraud >= threshold

    st.divider()
    if is_fraud:
        st.error(f"⚠️ Likely FRAUDULENT — {prob_fraud:.1%} confidence")
    else:
        st.success(f"✅ Likely LEGITIMATE — {(1 - prob_fraud):.1%} confidence")

    # ---------- SHAP explanation: top contributing words ----------
    shap_values = explainer(vec)
    feature_names = np.array(tfidf.get_feature_names_out())
    contributions = shap_values.values[0]

    # Only look at non-zero TF-IDF features present in this posting
    nonzero_idx = vec.nonzero()[1]
    pairs = [(feature_names[i], contributions[i]) for i in nonzero_idx]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    st.subheader("Why this verdict")
    top_n = pairs[:8]
    if top_n:
        st.write("Top words/phrases driving this decision:")
        for word, val in top_n:
            direction = "→ pushes toward FRAUD" if val > 0 else "→ pushes toward LEGIT"
            st.write(f"- **{word}** {direction} (impact: {val:+.3f})")

    flags = check_red_flags(job_text)
    if flags:
        st.subheader("Additional red-flag phrases detected")
        for f in flags:
            st.write(f"- \"{f}\"")

    st.caption(
        "This tool gives a probability estimate based on patterns in historical "
        "job postings. Always verify independently (company registration, "
        "recruiter LinkedIn profile, official company domain) before acting."
    )