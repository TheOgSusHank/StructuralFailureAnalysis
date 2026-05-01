# Deploy as a Free Public Web App

The easiest free deployment path for this project is Streamlit Community Cloud.
It deploys directly from a GitHub repository and gives you a public URL on
`streamlit.app`.

## Files to Commit

Commit these files:

- `app.py`
- `requirements.txt`
- `data/generate_data.py`
- `data/__init__.py`
- `model/train.py`
- `model/train_image_rf.py`
- `README.md`

Do not commit these generated or large files:

- `data/raw/`
- `artifacts/`
- `__pycache__/`
- `*.log`

The `.gitignore` file is already set up for this.

## Deploy Steps

1. Create a GitHub repository for this project.
2. Push this project to GitHub.
3. Go to `https://share.streamlit.io`.
4. Sign in with GitHub.
5. Click `Create app`.
6. Choose the GitHub repository and branch.
7. Set the main file path to:

```text
app.py
```

8. Click `Deploy`.

Streamlit will install dependencies from `requirements.txt`, start `app.py`,
and give you a public URL.

## Local Test Before Deploying

Run this from the project folder:

```powershell
streamlit run app.py
```

If `streamlit` is not on your PATH, use:

```powershell
python -m streamlit run app.py
```

## Notes

The current public app predicts failure probability from:

- crack size
- stress intensity
- load cycles

It can use the generated Paris' Law dataset in `data/crack_growth_data.csv`,
but it also generates the same synthetic data at startup if the CSV is absent.
The concrete crack image dataset is useful for crack detection, but it is too
large to commit directly to a lightweight public Streamlit app.
