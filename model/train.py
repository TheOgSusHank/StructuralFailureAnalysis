from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "crack_growth_data.csv"


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)

    if "failure" not in data.columns:
        raise ValueError("Dataset must include a 'failure' target column.")

    features = data.drop(columns=["failure"])
    target = data["failure"]
    return features, target


def train_random_forest(features: pd.DataFrame, target: pd.Series) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest failure model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to CSV dataset. Defaults to {DEFAULT_DATASET}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, target = load_dataset(args.dataset)
    accuracy = train_random_forest(features, target)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
