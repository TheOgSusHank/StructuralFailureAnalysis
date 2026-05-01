from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data.generate_data import generate_crack_growth_data


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "crack_growth_data.csv"
FEATURE_COLUMNS = ["crack_length_mm", "stress_intensity", "load_cycles"]


st.set_page_config(page_title="Failure Risk Predictor", page_icon=":gear:", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return generate_crack_growth_data()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def train_model(path: Path) -> tuple[RandomForestClassifier, float, list[list[int]]]:
    data = load_dataset(path)
    x = data[FEATURE_COLUMNS]
    y = data["failure"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()
    return model, float(accuracy), matrix


def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.35:
        return "Moderate risk"
    return "Low risk"


st.title("Structural Failure Risk Predictor")

model, accuracy, matrix = train_model(DATASET_PATH)

left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.subheader("Inputs")
    crack_size = st.number_input(
        "Crack size (mm)",
        min_value=0.0,
        max_value=250.0,
        value=10.0,
        step=0.5,
    )
    stress = st.number_input(
        "Stress intensity",
        min_value=0.0,
        max_value=250.0,
        value=35.0,
        step=1.0,
    )
    cycles = st.number_input(
        "Load cycles",
        min_value=0,
        max_value=10_000_000,
        value=250_000,
        step=10_000,
    )

with right:
    input_row = pd.DataFrame(
        [
            {
                "crack_length_mm": crack_size,
                "stress_intensity": stress,
                "load_cycles": cycles,
            }
        ]
    )
    failure_probability = model.predict_proba(input_row)[0, 1]

    st.subheader("Prediction")
    st.metric("Failure probability", f"{failure_probability * 100:.1f}%", risk_label(failure_probability))
    st.progress(float(failure_probability))

st.divider()

metric_col, matrix_col = st.columns([0.6, 1.4], gap="large")

with metric_col:
    st.subheader("Model")
    st.metric("Accuracy", f"{accuracy:.3f}")
    if DATASET_PATH.exists():
        st.caption(f"Training data: {DATASET_PATH}")
    else:
        st.caption("Training data: generated at startup from Paris' Law")

with matrix_col:
    st.subheader("Confusion Matrix")
    confusion = pd.DataFrame(
        matrix,
        index=["Actual safe", "Actual failed"],
        columns=["Predicted safe", "Predicted failed"],
    )
    st.dataframe(confusion, use_container_width=True)
