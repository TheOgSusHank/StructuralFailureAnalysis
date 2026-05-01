from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MATERIAL_PROFILES: dict[str, dict[str, float]] = {
    "Aluminum 7075": {"toughness": 29.0, "fatigue_factor": 1.18},
    "Carbon Steel": {"toughness": 52.0, "fatigue_factor": 0.92},
    "Titanium Alloy": {"toughness": 66.0, "fatigue_factor": 0.78},
    "Stainless Steel": {"toughness": 82.0, "fatigue_factor": 0.70},
    "Composite": {"toughness": 44.0, "fatigue_factor": 1.05},
}

FEATURE_COLUMNS = ["crack_length_mm", "stress_intensity", "load_cycles", "material_type"]
MODEL_PATH = Path("artifacts/failure_model.joblib")


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    test_rows: int


def generate_synthetic_data(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create repeatable example data with realistic monotonic risk behavior."""
    rng = np.random.default_rng(random_state)
    materials = np.array(list(MATERIAL_PROFILES))

    material_type = rng.choice(materials, size=n_rows, p=[0.20, 0.25, 0.18, 0.22, 0.15])
    crack_length_mm = rng.gamma(shape=2.2, scale=5.0, size=n_rows).clip(0.2, 45.0)
    stress_intensity = rng.normal(loc=33.0, scale=12.0, size=n_rows).clip(5.0, 95.0)
    load_cycles = np.exp(rng.normal(loc=np.log(250_000), scale=1.25, size=n_rows)).clip(500, 8_000_000)

    toughness = np.array([MATERIAL_PROFILES[m]["toughness"] for m in material_type])
    fatigue_factor = np.array([MATERIAL_PROFILES[m]["fatigue_factor"] for m in material_type])

    crack_severity = np.sqrt(crack_length_mm / 12.0)
    stress_ratio = stress_intensity / toughness
    fatigue_severity = np.log10(load_cycles) / 6.2
    interaction = crack_severity * stress_ratio * fatigue_severity * fatigue_factor

    logit = -5.1 + 5.5 * stress_ratio + 1.25 * crack_severity + 2.9 * interaction
    true_probability = 1.0 / (1.0 + np.exp(-logit))
    failed = rng.binomial(1, true_probability)

    return pd.DataFrame(
        {
            "crack_length_mm": crack_length_mm.round(3),
            "stress_intensity": stress_intensity.round(3),
            "load_cycles": load_cycles.round().astype(int),
            "material_type": material_type,
            "failure": failed,
            "true_failure_probability": true_probability,
        }
    )


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric_features = ["crack_length_mm", "stress_intensity", "load_cycles"]
    categorical_features = ["material_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("material", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate(
    n_rows: int = 5000,
    random_state: int = 42,
    test_size: float = 0.25,
) -> tuple[Pipeline, EvaluationResult, pd.DataFrame]:
    data = generate_synthetic_data(n_rows=n_rows, random_state=random_state)
    x = data[FEATURE_COLUMNS]
    y = data["failure"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = build_pipeline(random_state=random_state)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = EvaluationResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        confusion_matrix=confusion_matrix(y_test, predictions).astype(int).tolist(),
        test_rows=len(x_test),
    )

    scored_test = x_test.copy()
    scored_test["actual_failure"] = y_test.to_numpy()
    scored_test["predicted_failure_probability"] = probabilities
    return model, metrics, scored_test


def predict_failure_probability(
    model: Pipeline,
    crack_length_mm: float,
    stress_intensity: float,
    load_cycles: int,
    material_type: str,
) -> float:
    row = pd.DataFrame(
        [
            {
                "crack_length_mm": crack_length_mm,
                "stress_intensity": stress_intensity,
                "load_cycles": load_cycles,
                "material_type": material_type,
            }
        ]
    )
    return float(model.predict_proba(row)[0, 1])


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline | None:
    if not path.exists():
        return None
    return joblib.load(path)
