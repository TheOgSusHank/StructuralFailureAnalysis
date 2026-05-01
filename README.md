# Structural Failure Risk Predictor

Predict failure probability from crack size, stress intensity, and load cycles using Python, scikit-learn, and Streamlit.

## Features

- Generate synthetic structural integrity data when no field data is available.
- Train a scikit-learn model for failure classification.
- Evaluate accuracy, ROC AUC, and confusion matrix.
- Use a Streamlit dashboard for training, evaluation, and live predictions.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/generate_data.py
python model/train.py
streamlit run app.py
```

The Streamlit app trains a Random Forest model at startup from `data/crack_growth_data.csv`.
If that CSV is not present, the app generates synthetic Paris' Law training data automatically.

## Deploy Online

See [DEPLOYMENT.md](DEPLOYMENT.md) for steps to publish this as a free public Streamlit web app.

## Inputs

- Crack length in millimeters
- Stress intensity in MPa*sqrt(m)
- Load cycles

## Output

- Failure probability from 0 to 100%

## Notes

This project uses synthetic data generated from Paris' Law. Replace the generated dataset with validated inspection, laboratory, or simulation data before using this for engineering decisions.
