import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Hard Drive Failure Predictor", layout="wide")

# ---- Load artifacts ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_rf_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    le = joblib.load("label_encoder.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, feature_cols, le, threshold

model, feature_cols, le, threshold = load_artifacts()

SMART_COLS = ['smart_1_raw', 'smart_2_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw',
              'smart_7_raw', 'smart_8_raw', 'smart_9_raw', 'smart_10_raw', 'smart_12_raw',
              'smart_183_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw',
              'smart_189_raw', 'smart_190_raw', 'smart_191_raw', 'smart_192_raw',
              'smart_193_raw', 'smart_194_raw', 'smart_196_raw', 'smart_197_raw',
              'smart_198_raw', 'smart_199_raw', 'smart_240_raw', 'smart_241_raw',
              'smart_242_raw']

ENGINEERED_COLS = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']

REQUIRED_COLS = ['date', 'serial_number', 'model'] + SMART_COLS


def engineer_features(df):
    """Replicates the notebook's preprocessing pipeline on raw multi-day SMART data."""
    df = df.copy()

    # dtype fixes
    df['date'] = pd.to_datetime(df['date'])
    for col in SMART_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[SMART_COLS] = df[SMART_COLS].fillna(0)

    # sort for correct groupby-based feature engineering
    df = df.sort_values(by=['serial_number', 'date']).reset_index(drop=True)

    # drive age
    df['drive_age_days'] = df.groupby('serial_number')['date'].transform(
        lambda date: (date - date.min()).dt.days
    )

    # delta + rolling average features
    for col in ENGINEERED_COLS:
        df[f'{col}_delta'] = df.groupby('serial_number')[col].transform(lambda x: x.diff())
        df[f'{col}_delta'] = df[f'{col}_delta'].fillna(0)
        df[f'{col}_roll_avg'] = df.groupby('serial_number')[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

    # model encoding — drop rows with unseen model strings
    known_models = set(le.classes_)
    unseen_mask = ~df['model'].isin(known_models)
    n_unseen = unseen_mask.sum()
    if n_unseen > 0:
        st.warning(
            f"Dropped {n_unseen} row(s) with a drive `model` not seen during training "
            f"(cannot be scored reliably): {sorted(df.loc[unseen_mask, 'model'].unique())}"
        )
        df = df.loc[~unseen_mask].reset_index(drop=True)

    if df.empty:
        return df

    df['model_encoded'] = le.transform(df['model'])

    return df


st.title("Hard Drive Failure Predictor")
st.write(
    "Upload multi-day SMART telemetry (raw Backblaze-style format: one row per "
    "drive per day) to predict which drives are at risk of failing within 30 days."
)

with st.expander("Required CSV columns"):
    st.code(", ".join(REQUIRED_COLS))
    st.caption(
        "Each drive should have multiple rows (different dates) so that delta and "
        "rolling-average features can be computed. A single-day snapshot per drive "
        "will still work, but those features default to 0 for the first observed day."
    )

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in REQUIRED_COLS if c not in raw_df.columns]
    if missing_cols:
        st.error(f"Missing required column(s): {missing_cols}")
    else:
        with st.spinner("Engineering features and scoring drives..."):
            processed = engineer_features(raw_df)

        if processed.empty:
            st.error("No valid rows remaining after preprocessing.")
        else:
            X = processed[feature_cols]
            probs = model.predict_proba(X)[:, 1]
            processed['failure_probability'] = probs
            processed['flagged'] = (probs >= threshold).astype(int)

            n_flagged = processed['flagged'].sum()
            st.success(f"Scored {len(processed)} drive-day rows — {n_flagged} flagged at threshold {threshold}.")

            result_cols = ['serial_number', 'date', 'model', 'failure_probability', 'flagged']
            display_df = processed[result_cols].sort_values('failure_probability', ascending=False)
            st.dataframe(display_df, use_container_width=True)

            csv_out = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download results as CSV", csv_out, "predictions.csv", "text/csv")
