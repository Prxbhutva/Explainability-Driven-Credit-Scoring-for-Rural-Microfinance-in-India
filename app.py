# app.py
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ========== setup ==========
st.set_page_config(page_title="Rural Credit Scoring", page_icon="💳", layout="wide")
st.title("💳 Explainability-Driven Credit Scoring (Rural Microfinance)")
warnings.filterwarnings("ignore", message="If you are loading a serialized model")

MODEL_PATH = Path("RuralCreditModel.joblib")
FEATS_PATH = Path("feature_columns.json")

if not MODEL_PATH.exists():
    st.error("❌ Missing RuralCreditModel.joblib next to app.py")
    st.stop()
pipe = load(MODEL_PATH)

# ========== feature order from model ==========
def expected_from_model(pipeline):
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return None

EXPECTED_COLS = expected_from_model(pipe)
if EXPECTED_COLS is None:
    if FEATS_PATH.exists():
        EXPECTED_COLS = json.loads(FEATS_PATH.read_text())
        st.warning("Could not read feature names from model; using feature_columns.json fallback.")
    else:
        st.error("❌ Could not discover feature names and no feature_columns.json provided.")
        st.stop()
else:
    st.success(f"✅ Discovered {len(EXPECTED_COLS)} feature names from the model.")

# ========== helpers ==========
def group_suffixes(prefix, cols):
    pref = prefix + "_"
    return sorted({c[len(pref):] for c in cols if c.startswith(pref)})

def set_one_hot(prefix, choice, frame):
    pref = prefix + "_"
    for c in frame.columns:
        if c.startswith(pref):
            frame.at[0, c] = 1 if c == f"{pref}{choice}" else 0

def bool01(x):
    x = str(x).strip().lower()
    return 1 if x in ("yes", "y", "true", "1", "own", "owned") else 0

import re

def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    def clean_cell(x):
        if pd.isna(x):
            return 0.0
        s = str(x).strip()
        # Remove brackets and commas
        s = re.sub(r"[\[\],]", "", s)
        # Handle weird scientific notation like 5E-1 or [5E-1]
        try:
            return float(s)
        except ValueError:
            try:
                return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)[0])
            except Exception:
                return 0.0
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(clean_cell)
    return out.astype(np.float32)



def metric_table(pd_value, threshold, label):
    df = pd.DataFrame({
        "Metric": ["PD (probability)", "Threshold", "Predicted label"],
        "Value": [f"{pd_value:.3f}", f"{threshold:.3f}", str(int(label))]
    })
    return df

# ========== base one-row ==========
X = pd.DataFrame(np.zeros((1, len(EXPECTED_COLS))), columns=EXPECTED_COLS, dtype=np.float32)

groups = {
    "social_class": group_suffixes("social_class", EXPECTED_COLS),
    "city": group_suffixes("city", EXPECTED_COLS),
    "type_of_house": group_suffixes("type_of_house", EXPECTED_COLS),
    "loan_purpose": group_suffixes("loan_purpose", EXPECTED_COLS),
    "primary_business": group_suffixes("primary_business", EXPECTED_COLS),
    "secondary_business": group_suffixes("secondary_business", EXPECTED_COLS),
}

# ===== UI =====
st.header("1) Demographics")
c1, c2 = st.columns(2)
with c1:
    sex_choice = st.selectbox("Sex", ["male", "female"], index=0)
with c2:
    sc_choice = st.selectbox("Social Class", options=groups["social_class"] or ["unknown"])
if "sex" in X.columns:
    X.at[0, "sex"] = 1 if sex_choice == "male" else 0
if groups["social_class"]:
    set_one_hot("social_class", sc_choice, X)

st.header("2) Household")
h1, h2, h3 = st.columns(3)
with h1:
    home_ownership = st.selectbox("Home Ownership", ["yes", "no"])
    occupants_count = st.number_input("Occupants Count", 1, 20, 4)
with h2:
    house_area = st.number_input("House Area (sq. ft)", 40, 3000, 60)
    sanitary_availability = st.selectbox("Sanitary Available", ["yes", "no"])
with h3:
    water_availabity = st.selectbox("Water Available", ["yes", "no"])
    toh_choice = st.selectbox("Type of House", options=groups["type_of_house"] or ["t1", "t2"])
for name, val in [
    ("home_ownership", bool01(home_ownership)),
    ("occupants_count", occupants_count),
    ("house_area", house_area),
    ("sanitary_availability", bool01(sanitary_availability)),
    ("water_availabity", bool01(water_availabity)),
]:
    if name in X.columns:
        X.at[0, name] = val
if groups["type_of_house"]:
    set_one_hot("type_of_house", toh_choice, X)

st.header("3) Financials")
f1, f2, f3 = st.columns(3)
with f1:
    annual_income = st.number_input("Annual Income (₹)", 0, 2_000_000, 24000)
with f2:
    monthly_income = st.number_input("Monthly Income (₹)", 0, 200_000, 2000)
with f3:
    estimated_savings = st.number_input("Estimated Savings (₹/year)", 0, 1_000_000, 5000)
d1, d2, d3 = st.columns(3)
with d1:
    old_dependents = st.number_input("Old Dependents", 0, 10, 1)
with d2:
    young_dependents = st.number_input("Young Dependents", 0, 10, 2)
with d3:
    agri_loan_flag = st.selectbox("Agricultural Loan", ["no", "yes"])
for name, val in [
    ("annual_income", annual_income),
    ("monthly_income", monthly_income),
    ("estimated_savings", estimated_savings),
    ("old_dependents", old_dependents),
    ("young_dependents", young_dependents),
    ("agri_loan", 1 if agri_loan_flag == "yes" else 0),
]:
    if name in X.columns:
        X.at[0, name] = val

st.header("4) Location")
city_choice = st.selectbox("City / Village", options=groups["city"] or ["unknown"])
if groups["city"]:
    set_one_hot("city", city_choice, X)

st.header("5) Occupation")
o1, o2 = st.columns(2)
with o1:
    primary_choice = st.selectbox("Primary Business", options=groups["primary_business"] or ["unknown"])
with o2:
    secondary_choice = st.selectbox("Secondary Business", options=groups["secondary_business"] or ["none"])
if groups["primary_business"]:
    set_one_hot("primary_business", primary_choice, X)
if groups["secondary_business"]:
    set_one_hot("secondary_business", secondary_choice, X)

st.header("6) Loan Purpose")
lp_choice = st.selectbox("Loan Purpose", options=groups["loan_purpose"] or ["education loan"])
if groups["loan_purpose"]:
    set_one_hot("loan_purpose", lp_choice, X)

# sanitize
X = sanitize_numeric(X)

with st.expander("Preview non-zero features in your input row"):
    nz = X.loc[:, (X != 0).any()].T
    st.dataframe(nz)

# ===== Predict =====
st.markdown("---")
st.header("7) Predict Default Risk")

# Fixed threshold (set by risk calibration)
threshold = 0.18
st.info(f"Decision threshold fixed at **{threshold:.2f}** (based on risk calibration by institution).")

if st.button("Predict"):
    # PD
    if hasattr(pipe, "predict_proba"):
        pd_value = float(pipe.predict_proba(X)[:, 1][0])
    else:
        pd_value = float(pipe.predict(X)[0])

    label = int(pd_value >= threshold)

    # BIG banner
    if label == 1:
        st.error(f"🔥 Predicted label: **1 (Default)** — PD = {pd_value:.3f}")
    else:
        st.success(f"✅ Predicted label: **0 (No Default)** — PD = {pd_value:.3f}")

    # Summary table
    st.dataframe(metric_table(pd_value, threshold, label))

    # ===== SHAP Explainability =====
    # ===== SHAP Explainability =====
# ===== EXPLAIN: XGBoost contributions (TreeSHAP via pred_contribs) =====
with st.expander("🔍 Explain this prediction (XGBoost contributions)"):
    try:
        import xgboost as xgb

        model = getattr(pipe, "named_steps", {}).get("model", pipe)
        X_np = X.values.astype(np.float32)

        dmat = xgb.DMatrix(X_np, feature_names=EXPECTED_COLS)
        booster = getattr(model, "get_booster", lambda: None)()

        if booster is None:
            st.info("Booster not available; cannot compute contributions.")
        else:
            contrib = booster.predict(dmat, pred_contribs=True)  # shape = (1, n_features+1)
            contrib = contrib[0]
            base = float(contrib[-1])
            phi = contrib[:-1]

            idx = np.argsort(-np.abs(phi))
            top_k = min(20, len(phi))
            top = [(EXPECTED_COLS[i], float(phi[i])) for i in idx[:top_k]]

            imp_df = pd.DataFrame(top, columns=["Feature", "Contribution (log-odds)"])
            imp_df["|Contribution|"] = imp_df["Contribution (log-odds)"].abs()

            st.write("Top feature contributions (log-odds scale):")
            st.dataframe(imp_df)

            st.caption(
                f"Base (bias) log-odds: {base:.4f} | "
                f"Sum contrib: {phi.sum():.4f} | "
                f"Pred log-odds: {base + phi.sum():.4f} | "
                f"Reconstructed PD: {1 / (1 + np.exp(-(base + phi.sum()))):.4f}"
            )

    except Exception as e:
        st.warning(f"Could not compute XGBoost contributions: {e}")
