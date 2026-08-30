"""
Generate exploratory data analysis visuals for the cleaned job postings
dataset — class balance, missingness-as-signal, text length patterns,
and categorical breakdowns. Saves each chart as a PNG for your report.

Run locally:
    pip install matplotlib seaborn pandas
    python dataset_insights.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Paths resolved relative to THIS script's location ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "processed_data", "preprocessed_cleaned.csv")  # adjust if needed
OUT_DIR = os.path.join(SCRIPT_DIR, "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {"legit": "#3fa796", "fraud": "#c1443b"}
FRAUD_LABELS = ["Legitimate", "Fraudulent"]

df = pd.read_csv(DATA_PATH)
df["label"] = df["fraudulent"].map({0: "Legitimate", 1: "Fraudulent"})
print(f"Loaded {len(df)} rows.\n")

# ==================================================================
# 1. Class balance
# ==================================================================
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["label"].value_counts()
bars = ax.bar(counts.index, counts.values, color=[PALETTE["legit"], PALETTE["fraud"]])
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 100, f"{h:,}\n({h/len(df)*100:.1f}%)",
            ha="center", fontsize=11)
ax.set_title("Class Balance: Legitimate vs Fraudulent Postings", fontsize=13, weight="bold")
ax.set_ylabel("Number of postings")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_class_balance.png"), dpi=150)
plt.close()
print("Saved 01_class_balance.png")

# ==================================================================
# 2. Missingness as a fraud signal (the key finding from EDA)
# ==================================================================
missing_cols = ["company_profile", "requirements", "industry", "location"]
fraud_rates = []
for col in missing_cols:
    flag_col = f"{col}_missing"
    if flag_col not in df.columns:
        df[flag_col] = df[col].isna().astype(int)
    rate_present = df[df[flag_col] == 0]["fraudulent"].mean() * 100
    rate_missing = df[df[flag_col] == 1]["fraudulent"].mean() * 100
    fraud_rates.append((col, rate_present, rate_missing))

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(missing_cols))
width = 0.35
present_vals = [r[1] for r in fraud_rates]
missing_vals = [r[2] for r in fraud_rates]
ax.bar([i - width/2 for i in x], present_vals, width, label="Field present", color=PALETTE["legit"])
ax.bar([i + width/2 for i in x], missing_vals, width, label="Field missing", color=PALETTE["fraud"])
ax.set_xticks(list(x))
ax.set_xticklabels([c.replace("_", " ").title() for c in missing_cols])
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud Rate by Field Missingness", fontsize=13, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_missingness_fraud_rate.png"), dpi=150)
plt.close()
print("Saved 02_missingness_fraud_rate.png")

# ==================================================================
# 3. Text length distribution by class
# ==================================================================
df["text_length"] = df["combined_text"].astype(str).str.len()
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=df[df["fraudulent"] == 0], x="text_length", fill=True,
            color=PALETTE["legit"], label="Legitimate", ax=ax, clip=(0, df["text_length"].quantile(0.98)))
sns.kdeplot(data=df[df["fraudulent"] == 1], x="text_length", fill=True,
            color=PALETTE["fraud"], label="Fraudulent", ax=ax, clip=(0, df["text_length"].quantile(0.98)))
ax.set_title("Posting Text Length by Class", fontsize=13, weight="bold")
ax.set_xlabel("Character count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_text_length_distribution.png"), dpi=150)
plt.close()
print("Saved 03_text_length_distribution.png")

# ==================================================================
# 4. Categorical breakdowns: employment_type, required_experience, required_education
# ==================================================================
cat_cols = [c for c in ["employment_type", "required_experience", "required_education"] if c in df.columns]
for col in cat_cols:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, hue="label",
                  palette=PALETTE.values(), ax=ax)
    ax.set_title(f"{col.replace('_', ' ').title()} — Volume by Class", fontsize=13, weight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.legend(title="")
    plt.tight_layout()
    fname = f"04_{col}_breakdown.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")

# ==================================================================
# 5. Fraud rate (not just volume) by employment type
# ==================================================================
if "employment_type" in df.columns:
    rate_by_type = df.groupby("employment_type")["fraudulent"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    rate_by_type.plot(kind="barh", ax=ax, color=PALETTE["fraud"])
    ax.set_title("Fraud Rate by Employment Type", fontsize=13, weight="bold")
    ax.set_xlabel("Fraud rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_fraud_rate_by_employment_type.png"), dpi=150)
    plt.close()
    print("Saved 05_fraud_rate_by_employment_type.png")

# ==================================================================
# 6. Correlation heatmap of numeric/flag features vs fraudulent
# ==================================================================
numeric_cols = [c for c in ["telecommuting", "has_company_logo", "has_questions",
                             "company_profile_missing", "requirements_missing",
                             "industry_missing", "location_missing", "fraudulent"]
                if c in df.columns]
if len(numeric_cols) > 2:
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Between Flags/Metadata and Fraud", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved 06_correlation_heatmap.png")

# ==================================================================
# 7. Missing value summary (overall, before flags)
# ==================================================================
missing_pct = df[missing_cols].isna().mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 4))
missing_pct.plot(kind="barh", ax=ax, color="#8993a3")
ax.set_title("Missing Data by Field", fontsize=13, weight="bold")
ax.set_xlabel("Missing (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_missing_data_summary.png"), dpi=150)
plt.close()
print("Saved 07_missing_data_summary.png")

print(f"\nAll charts saved to: {OUT_DIR}")