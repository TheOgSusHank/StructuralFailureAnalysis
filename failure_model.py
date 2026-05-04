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
    "Aluminum 7075": {"toughness": 29.0, "fatigue_factor": 1.18, "paris_c": 1.5e-12, "paris_m": 3.1},
    "Carbon Steel": {"toughness": 52.0, "fatigue_factor": 0.92, "paris_c": 2.5e-12, "paris_m": 2.9},
    "Titanium Alloy": {"toughness": 66.0, "fatigue_factor": 0.78, "paris_c": 0.8e-12, "paris_m": 3.0},
    "Stainless Steel": {"toughness": 82.0, "fatigue_factor": 0.70, "paris_c": 1.2e-12, "paris_m": 2.8},
    "Composite": {"toughness": 44.0, "fatigue_factor": 1.05, "paris_c": 3.0e-12, "paris_m": 3.2},
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
    """Create repeatable example data with physics-informed behavior using Paris' Law."""
    rng = np.random.default_rng(random_state)
    materials = np.array(list(MATERIAL_PROFILES))
    material_type = rng.choice(materials, size=n_rows)
    
    # Material-specific parameters
    toughness = np.array([MATERIAL_PROFILES[m]["toughness"] for m in material_type])
    paris_c_base = np.array([MATERIAL_PROFILES[m]["paris_c"] for m in material_type])
    paris_m_base = np.array([MATERIAL_PROFILES[m]["paris_m"] for m in material_type])
    
    initial_crack_m = rng.uniform(0.0005, 0.012, n_rows)
    stress_range_mpa = rng.uniform(25.0, 180.0, n_rows)
    load_cycles = rng.integers(1_000, 2_000_000, n_rows)
    geometry_factor = rng.normal(1.12, 0.06, n_rows).clip(1.0, 1.35)
    
    # Add some noise to material constants
    paris_c = paris_c_base * np.exp(rng.normal(0, 0.1, n_rows))
    paris_m = paris_m_base + rng.normal(0, 0.05, n_rows)
    
    delta_k = geometry_factor * stress_range_mpa * np.sqrt(np.pi * initial_crack_m)
    crack_growth_m = paris_c * np.power(delta_k, paris_m) * load_cycles
    crack_length_m = initial_crack_m + crack_growth_m
    
    final_stress_intensity = geometry_factor * stress_range_mpa * np.sqrt(np.pi * crack_length_m)
    
    # Failure if crack exceeds critical size or stress intensity exceeds toughness
    critical_crack_m = np.power(toughness / (geometry_factor * stress_range_mpa), 2) / np.pi
    failure = ((crack_length_m >= critical_crack_m) | (final_stress_intensity >= toughness)).astype(int)
    
    return pd.DataFrame({
        "crack_length_mm": (crack_length_m * 1000.0).round(3),
        "stress_intensity": final_stress_intensity.round(3),
        "load_cycles": load_cycles,
        "material_type": material_type,
        "failure": failure
    })

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
        x, y, test_size=test_size, stratify=y, random_state=random_state,
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
    row = pd.DataFrame([{
        "crack_length_mm": crack_length_mm,
        "stress_intensity": stress_intensity,
        "load_cycles": load_cycles,
        "material_type": material_type,
    }])
    return float(model.predict_proba(row)[0, 1])

def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path = MODEL_PATH) -> Pipeline | None:
    if not path.exists():
        return None
    return joblib.load(path)
