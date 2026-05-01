from __future__ import annotations

from failure_model import MODEL_PATH, save_model, train_and_evaluate


def main() -> None:
    model, metrics, _ = train_and_evaluate(n_rows=6000, random_state=42)
    save_model(model, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"ROC AUC: {metrics.roc_auc:.3f}")
    print(f"Confusion matrix: {metrics.confusion_matrix}")


if __name__ == "__main__":
    main()
