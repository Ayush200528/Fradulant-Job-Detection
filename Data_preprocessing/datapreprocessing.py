import pandas as pd
import numpy as np
import re
import os

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('fake_job_postings.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Drop irrelevant columns ────────────────────────────────────────────────────
DROP_COLS = ['job_id', 'department', 'salary_range', 'benefits', 'function']
df.drop(columns=DROP_COLS, inplace=True)
print(f"After dropping cols: {df.shape}")

# ── Fill missing text columns with empty string ────────────────────────────────
TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'location', 'industry']
for col in TEXT_COLS:
    df[col] = df[col].fillna('')

# ── Fill missing categorical columns with 'Unknown' ───────────────────────────
CAT_COLS = ['employment_type', 'required_experience', 'required_education']
for col in CAT_COLS:
    df[col] = df[col].fillna('Unknown')

# ── Combine all text columns into one for BERT ────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)        # remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # collapse whitespace
    return text

df['combined_text'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements']
).apply(clean_text)

print(f"Sample combined text:\n{df['combined_text'].iloc[0][:200]}\n")

# ── Structured features ────────────────────────────────────────────────────────
# Binary flags are already 0/1: telecommuting, has_company_logo, has_questions
# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

le_emp  = LabelEncoder()
le_exp  = LabelEncoder()
le_edu  = LabelEncoder()

df['employment_type_enc']    = le_emp.fit_transform(df['employment_type'])
df['required_experience_enc']= le_exp.fit_transform(df['required_experience'])
df['required_education_enc'] = le_edu.fit_transform(df['required_education'])

# ── Final structured feature columns ─────────────────────────────────────────
STRUCTURED_COLS = [
    'telecommuting', 'has_company_logo', 'has_questions',
    'employment_type_enc', 'required_experience_enc', 'required_education_enc'
]

# ── Save preprocessed data ────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
df.to_csv('data/preprocessed.csv', index=False)
print(f"Saved preprocessed data → data/preprocessed.csv")
print(f"Final shape: {df.shape}")
print(f"\nClass distribution:\n{df['fraudulent'].value_counts()}")
print(f"\nStructured features: {STRUCTURED_COLS}")
print(f"Text feature: combined_text")
print(f"Target: fraudulent")