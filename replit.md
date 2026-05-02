# Structural Failure Risk Predictor

## Overview

A machine learning web application that estimates the probability of failure in fatigue-loaded structural components. It uses crack size, stress intensity, and accumulated load cycles to predict failure risk via a Random Forest model trained on synthetic data derived from Paris' Law.

## Tech Stack

- **Language**: Python 3.x
- **Frontend/UI**: Streamlit
- **Machine Learning**: scikit-learn (RandomForestClassifier)
- **Data**: pandas, numpy
- **Visualization**: matplotlib
- **Model Persistence**: joblib

## Project Structure

```
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── data/
│   ├── __init__.py
│   └── generate_data.py    # Synthetic crack growth data generation (Paris' Law)
├── model/
│   ├── train.py            # Random Forest training logic
│   └── train_image_rf.py   # Extended training script
├── failure_model.py        # Standalone model utility
├── train_model.py          # Standalone training script
└── README.md / DEPLOYMENT.md
```

## Running the App

The app runs via Streamlit on port 5000:

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Key Features

- **Risk Gauge**: Half-circle gauge showing failure probability
- **KPI Dashboard**: Failure probability %, risk classification, model accuracy, training rows
- **Model Performance Tab**: Confusion matrix and feature importance charts
- **Sidebar Inputs**: Crack size (mm), stress intensity, load cycles
- **Risk Thresholds**: Low (<35%), Moderate (35-70%), High (≥70%)

## Data & Model

- Data is auto-generated at startup if `data/crack_growth_data.csv` doesn't exist
- RandomForestClassifier with 200 estimators, balanced class weights
- Features: crack_length_mm, stress_intensity, load_cycles
- Target: binary failure label

## Deployment

- **Target**: Autoscale
- **Run Command**: `streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true`
