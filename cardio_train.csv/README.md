# CardioRisk Classifier

CardioRisk Classifier is a Flask-based cardiovascular risk classification project. It combines feature ranking, gradient boosting, and isotonic probability calibration in a web interface that accepts standard clinical values.

## Highlights

- Calibrated cardiovascular risk prediction
- Patient form with readable medical inputs
- Holdout metrics: accuracy, precision, recall, F1, ROC AUC, and Brier score
- Feature ranking with correlation, mutual information, and chi-square methods
- Model analysis charts for feature importance and confusion matrix
- Responsive dashboard suitable for academic demonstrations and presentations

## Method

The app uses the UCI Cleveland Heart Disease dataset when available. The original disease severity target is converted to a binary risk label:

- `0`: low risk / no diagnosed disease
- `1-4`: high risk / disease present

The model is trained as an sklearn pipeline:

1. Raw clinical features
2. Standard scaling
3. Gradient boosting classifier
4. Isotonic calibration using `CalibratedClassifierCV`

If the UCI dataset cannot be downloaded, the app falls back to a deterministic offline demonstration dataset so the interface still works during presentations.

## Project Structure

```text
.
|-- app.py                 # Flask API and ML pipeline
|-- cardio_project.py      # Original research script
|-- requirements.txt       # Python dependencies
|-- run.bat                # Windows launcher
|-- test_app.py            # Smoke test script
`-- templates/
    `-- index.html         # Web dashboard
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

On Windows, you can also double-click `run.bat`.

## Test

```bash
python test_app.py
```

The smoke test checks:

- `/api/health`
- `/api/initialize`
- `/api/predict`
- `/api/feature-importance`
- `/api/confusion-matrix`

## Notes

This project is for educational and research demonstration purposes. It should not be used as a medical diagnosis tool.
