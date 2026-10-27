"""
Streamlit dashboard for Explainable and Uncertainty-Aware Sensor Data Detection.
Run: streamlit run app.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import make_features, generate_dataset, FEATURE_COLUMNS, BASE_COLUMNS

MODEL_PATH = Path("models/sensor_models.joblib")
DATA_PATH = Path("data/sensor_data.csv")

st.set_page_config(page_title="Explainable Sensor Data Detection", page_icon="📡", layout="wide")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def prepare_upload(df):
    required = set(BASE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if "timestamp" not in df.columns:
        df = df.copy()
        df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="min")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in BASE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def simple_explanation(row):
    reasons = []
    if abs(row.get("soil_moisture_change", 0)) > 8:
        reasons.append("large change in soil moisture compared with the previous reading")
    if row.get("temperature_peer_diff", 0) > 5:
        reasons.append("temperature differs substantially from the peer sensor")
    if row.get("soil_peer_diff", 0) > 8:
        reasons.append("soil moisture differs substantially from the peer sensor")
    if row.get("soil_frozen", 0) == 1:
        reasons.append("soil-moisture reading appears frozen over several readings")
    if not reasons:
        reasons.append("the model found a pattern that differs from its learned normal behavior")
    return "; ".join(reasons)


def main():
    st.title("📡 Explainable AI for Fake Sensor Data Detection")
    st.caption("Temporal analysis + cross-sensor consistency + uncertainty-aware alerts")

    artifacts = load_artifacts()
    if artifacts is None:
        st.error("Model files are missing. Run `python generate_data.py` and `python train_model.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Data source")
        source = st.radio("Choose input", ["Use sample simulation", "Upload CSV"])
        st.markdown("---")
        st.subheader("Decision thresholds")
        suspicious_threshold = st.slider("Suspicious threshold", 0.50, 0.95, 0.65, 0.05)
        review_threshold = st.slider("Review lower threshold", 0.05, 0.49, 0.35, 0.05)
        st.caption("Thresholds are adjustable demonstration settings. Tune them using validation data before real deployment.")

    if source == "Use sample simulation":
        if DATA_PATH.exists():
            raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        else:
            raw = generate_dataset()
        raw = prepare_upload(raw)
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with these columns: " + ", ".join(BASE_COLUMNS))
            st.stop()
        try:
            raw = prepare_upload(pd.read_csv(uploaded))
        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")
            st.stop()

    featured = make_features(raw)
    X = featured[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    model = artifacts["model"]
    probability = model.predict_proba(X)[:, 1]

    # The probability is calibrated on the training distribution, but can still be
    # unreliable on a new dataset or under distribution shift.
    status = np.where(
        probability >= suspicious_threshold, "Suspicious",
        np.where(probability < review_threshold, "Likely normal", "Needs review")
    )
    featured["manipulation_score"] = probability
    featured["status"] = status
    featured["explanation"] = featured.apply(simple_explanation, axis=1)

    # Optional unsupervised score; shown as a separate signal, not a fake probability.
    try:
        iso_raw = artifacts["isolation_forest"].decision_function(X)
        featured["isolation_score"] = iso_raw
    except Exception:
        featured["isolation_score"] = np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows analyzed", f"{len(featured):,}")
    c2.metric("Suspicious", int((featured["status"] == "Suspicious").sum()))
    c3.metric("Needs review", int((featured["status"] == "Needs review").sum()))
    c4.metric("Likely normal", int((featured["status"] == "Likely normal").sum()))

    st.subheader("Sensor readings over time")
    sensor_choice = st.selectbox(
        "Choose a sensor",
        ["soil_moisture", "temperature", "humidity", "temperature_peer", "soil_moisture_peer"]
    )
    fig = px.line(featured, x="timestamp", y=sensor_choice, title=f"{sensor_choice.replace('_', ' ').title()} over time")
    suspicious_rows = featured[featured["status"] == "Suspicious"]
    if not suspicious_rows.empty:
        fig.add_scatter(
            x=suspicious_rows["timestamp"],
            y=suspicious_rows[sensor_choice],
            mode="markers",
            name="Suspicious",
            marker={"size": 9, "symbol": "x"},
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detection results")
    show_cols = [
        "timestamp", *BASE_COLUMNS, "manipulation_score",
        "status", "explanation"
    ]
    if "label" in featured.columns:
        show_cols.insert(1, "label")
    st.dataframe(featured[show_cols].tail(500), use_container_width=True, hide_index=True)

    st.subheader("Inspect one reading")
    if len(featured):
        selected_index = st.number_input(
            "Row index",
            min_value=0,
            max_value=len(featured) - 1,
            value=0,
            step=1,
        )
        row = featured.iloc[int(selected_index)]
        left, right = st.columns(2)
        with left:
            st.write("**Status:**", row["status"])
            st.write("**Manipulation score:**", f"{row['manipulation_score']:.3f}")
            st.write("**Explanation:**", row["explanation"])
        with right:
            st.write("**Temporal evidence**")
            st.write(f"Soil moisture change: {row['soil_moisture_change']:.3f}")
            st.write(f"Soil rolling standard deviation: {row['soil_rolling_std']:.3f}")
            st.write("**Cross-sensor evidence**")
            st.write(f"Temperature peer difference: {row['temperature_peer_diff']:.3f}")
            st.write(f"Soil peer difference: {row['soil_peer_diff']:.3f}")

        # SHAP explanation: supported when the installed SHAP version and model
        # wrapper work together. If not, the dashboard still provides rule-based
        # evidence and continues operating.
        with st.expander("Optional SHAP feature explanation"):
            try:
                import shap
                # CalibratedClassifierCV wraps the estimator. Use the first fitted
                # calibrated pipeline as a representative explanation model.
                calibrated = model.calibrated_classifiers_[0].estimator
                imputer = calibrated.named_steps["imputer"]
                forest = calibrated.named_steps["classifier"]
                sample = X.iloc[[int(selected_index)]]
                sample_imp = imputer.transform(sample)
                explainer = shap.TreeExplainer(forest)
                values = explainer.shap_values(sample_imp)
                if isinstance(values, list):
                    vals = np.asarray(values[-1])[0]
                else:
                    arr = np.asarray(values)
                    if arr.ndim == 3:
                        vals = arr[0, :, -1]
                    else:
                        vals = arr[0]
                shap_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "contribution": vals})
                shap_df["abs_contribution"] = shap_df["contribution"].abs()
                shap_df = shap_df.sort_values("abs_contribution", ascending=True).tail(10)
                shap_fig = px.bar(
                    shap_df, x="contribution", y="feature", orientation="h",
                    title="Top feature contributions (base Random Forest; not calibrated probability)",
                )
                st.plotly_chart(shap_fig, use_container_width=True)
                st.caption("SHAP values explain the underlying Random Forest output, not necessarily the calibrated probability. Interpret them alongside the model output.")
            except Exception as exc:
                st.warning(f"SHAP explanation was unavailable in this environment: {exc}")
                st.info("The temporal and cross-sensor evidence above is still available.")

    st.subheader("Download results")
    csv_bytes = featured.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analyzed CSV",
        data=csv_bytes,
        file_name="sensor_detection_results.csv",
        mime="text/csv",
    )

    if "label" in featured.columns:
        st.subheader("Evaluation on labeled data")
        st.caption("Only use these metrics when the labels are trustworthy. The uploaded labels are not used to make predictions.")
        from sklearn.metrics import classification_report, confusion_matrix
        y_true = pd.to_numeric(featured["label"], errors="coerce")
        valid = y_true.isin([0, 1])
        if valid.any():
            pred_label = (featured.loc[valid, "manipulation_score"] >= suspicious_threshold).astype(int)
            report = classification_report(
                y_true[valid].astype(int), pred_label,
                labels=[0, 1], target_names=["normal", "manipulated"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            st.write("Confusion matrix (rows=true labels, columns=predicted labels; order normal, manipulated):")
            st.write(confusion_matrix(y_true[valid].astype(int), pred_label, labels=[0, 1]))


if __name__ == "__main__":
    main()
