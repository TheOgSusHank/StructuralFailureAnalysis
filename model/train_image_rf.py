from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "concrete_images"
    / "2 classes(cracks and background)"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "concrete_crack_image_rf.joblib"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
IMAGE_SIZE = (32, 32)
CLASS_NAMES = ["Background", "Cracks"]


def build_feature_names() -> list[str]:
    pixel_names = [
        f"gray_pixel_row_{row}_col_{col}"
        for row in range(IMAGE_SIZE[1])
        for col in range(IMAGE_SIZE[0])
    ]
    color_names = [
        "red_mean",
        "green_mean",
        "blue_mean",
        "red_std",
        "green_std",
        "blue_std",
        "red_min",
        "green_min",
        "blue_min",
        "red_max",
        "green_max",
        "blue_max",
    ]
    texture_names = [
        "gray_mean",
        "gray_std",
        "edge_mean",
        "edge_std",
        "edge_90th_percentile",
    ]
    return pixel_names + color_names + texture_names


def extract_image_features(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    small = image.resize(IMAGE_SIZE)
    rgb = np.asarray(small, dtype=np.float32) / 255.0

    gray_image = small.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    edges = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0

    color_stats = np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            rgb.min(axis=(0, 1)),
            rgb.max(axis=(0, 1)),
        ]
    )
    texture_stats = np.array(
        [
            gray.mean(),
            gray.std(),
            edges.mean(),
            edges.std(),
            np.percentile(edges, 90),
        ],
        dtype=np.float32,
    )

    return np.concatenate([gray.ravel(), color_stats, texture_stats])


def load_image_dataset(image_root: Path) -> tuple[np.ndarray, np.ndarray]:
    class_map = {
        "Background": 0,
        "Cracks": 1,
    }
    features: list[np.ndarray] = []
    labels: list[int] = []

    for class_name, label in class_map.items():
        class_dir = image_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for image_path in sorted(class_dir.glob("*.jpg")):
            features.append(extract_image_features(image_path))
            labels.append(label)

    if not features:
        raise ValueError(f"No JPG images found under {image_root}")

    return np.vstack(features), np.asarray(labels)


def train_model(features: np.ndarray, labels: np.ndarray) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    return model, y_test, predictions


def save_confusion_matrix_plot(
    y_test: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Concrete Crack Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance_plot(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[index] for index in top_indices]
    top_scores = importances[top_indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_names[::-1], top_scores[::-1], color="#2f6f9f")
    ax.set_title(f"Top {top_n} Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest crack image classifier.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Folder containing Background and Cracks subfolders. Defaults to {DEFAULT_IMAGE_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the trained model. Defaults to {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory to save visualization PNGs. Defaults to {DEFAULT_PLOTS_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels = load_image_dataset(args.image_root)
    model, y_test, predictions = train_model(features, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    confusion_matrix_path = args.plots_dir / "confusion_matrix.png"
    feature_importance_path = args.plots_dir / "feature_importance.png"
    save_confusion_matrix_plot(y_test, predictions, confusion_matrix_path)
    save_feature_importance_plot(model, build_feature_names(), feature_importance_path)

    print(f"Images loaded: {len(labels):,}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))
    print(f"Saved model: {args.output}")
    print(f"Saved confusion matrix plot: {confusion_matrix_path}")
    print(f"Saved feature importance plot: {feature_importance_path}")


if __name__ == "__main__":
    main()
