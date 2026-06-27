from io import BytesIO
import base64
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, jsonify, render_template, request
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
CACHE_FILE = "heart_disease_cleveland.csv"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
COLUMN_NAMES = FEATURE_COLUMNS + ["target"]
FEATURE_DETAILS = [
    {"name": "age", "label": "Age", "type": "number", "min": 18, "max": 100, "step": 1, "value": 54},
    {"name": "sex", "label": "Sex", "type": "select", "value": 1, "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}]},
    {"name": "cp", "label": "Chest pain type", "type": "select", "value": 3, "options": [{"value": 1, "label": "Typical angina"}, {"value": 2, "label": "Atypical angina"}, {"value": 3, "label": "Non-anginal pain"}, {"value": 4, "label": "Asymptomatic"}]},
    {"name": "trestbps", "label": "Resting blood pressure", "type": "number", "min": 80, "max": 220, "step": 1, "value": 130, "unit": "mm Hg"},
    {"name": "chol", "label": "Serum cholesterol", "type": "number", "min": 100, "max": 650, "step": 1, "value": 240, "unit": "mg/dL"},
    {"name": "fbs", "label": "Fasting blood sugar", "type": "select", "value": 0, "options": [{"value": 0, "label": "<= 120 mg/dL"}, {"value": 1, "label": "> 120 mg/dL"}]},
    {"name": "restecg", "label": "Resting ECG", "type": "select", "value": 0, "options": [{"value": 0, "label": "Normal"}, {"value": 1, "label": "ST-T abnormality"}, {"value": 2, "label": "Left ventricular hypertrophy"}]},
    {"name": "thalach", "label": "Maximum heart rate", "type": "number", "min": 60, "max": 230, "step": 1, "value": 150},
    {"name": "exang", "label": "Exercise-induced angina", "type": "select", "value": 0, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    {"name": "oldpeak", "label": "ST depression", "type": "number", "min": 0, "max": 7, "step": 0.1, "value": 1.0},
    {"name": "slope", "label": "ST segment slope", "type": "select", "value": 2, "options": [{"value": 1, "label": "Upsloping"}, {"value": 2, "label": "Flat"}, {"value": 3, "label": "Downsloping"}]},
    {"name": "ca", "label": "Major vessels", "type": "select", "value": 0, "options": [{"value": 0, "label": "0"}, {"value": 1, "label": "1"}, {"value": 2, "label": "2"}, {"value": 3, "label": "3"}]},
    {"name": "thal", "label": "Thalassemia", "type": "select", "value": 3, "options": [{"value": 3, "label": "Normal"}, {"value": 6, "label": "Fixed defect"}, {"value": 7, "label": "Reversible defect"}]},
]

model = None
training_df = None
test_data = None
app_state = {
    "metrics": None,
    "rankings": None,
    "dataset": None,
}


def demo_dataset(seed=42, rows=360):
    """Deterministic fallback data for offline demos when UCI cannot be reached."""
    rng = np.random.default_rng(seed)
    age = rng.integers(34, 78, rows)
    sex = rng.binomial(1, 0.68, rows)
    cp = rng.choice([1, 2, 3, 4], rows, p=[0.12, 0.18, 0.25, 0.45])
    trestbps = np.clip(rng.normal(128 + (age - 52) * 0.35, 17), 90, 210).round()
    chol = np.clip(rng.normal(235 + (age - 52) * 1.2, 46), 130, 560).round()
    fbs = rng.binomial(1, np.clip((age - 35) / 110, 0.08, 0.42))
    restecg = rng.choice([0, 1, 2], rows, p=[0.52, 0.06, 0.42])
    thalach = np.clip(rng.normal(178 - age * 0.55, 18), 80, 205).round()
    exang = rng.binomial(1, np.where(cp == 4, 0.48, 0.18))
    oldpeak = np.clip(rng.gamma(1.6, 0.85, rows) + exang * 0.55, 0, 6.2).round(1)
    slope = rng.choice([1, 2, 3], rows, p=[0.45, 0.42, 0.13])
    ca = rng.choice([0, 1, 2, 3], rows, p=[0.58, 0.23, 0.13, 0.06])
    thal = rng.choice([3, 6, 7], rows, p=[0.56, 0.08, 0.36])
    score = (
        -6.0
        + 0.045 * age
        + 0.65 * sex
        + 0.85 * (cp == 4)
        + 0.015 * (trestbps - 120)
        + 0.006 * (chol - 200)
        - 0.028 * (thalach - 145)
        + 0.65 * exang
        + 0.55 * oldpeak
        + 0.42 * (slope == 2)
        + 0.72 * (slope == 3)
        + 0.55 * ca
        + 0.75 * (thal == 7)
    )
    probability = 1 / (1 + np.exp(-score))
    cardio = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "cardio": cardio,
        }
    )


def load_dataset():
    source = "UCI Heart Disease Cleveland dataset"
    try:
        df = pd.read_csv(CACHE_FILE)
    except Exception:
        try:
            df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES)
            df.to_csv(CACHE_FILE, index=False)
        except Exception:
            df = demo_dataset()
            source = "offline demonstration dataset"
    return df, source


def preprocess_data(df):
    df = df.copy()
    if "target" in df.columns:
        df = df.rename(columns={"target": "cardio"})

    for column in df.columns:
        df[column] = pd.to_numeric(df[column].replace("?", np.nan), errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["cardio"] = (df["cardio"] > 0).astype(int)
    return df[FEATURE_COLUMNS + ["cardio"]]


def base_estimator():
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    if LGB_AVAILABLE:
        return lgb.LGBMClassifier(random_state=42, n_estimators=140, verbose=-1)
    return GradientBoostingClassifier(random_state=42)


def train_model(df):
    global model, test_data

    X = df[FEATURE_COLUMNS]
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", base_estimator()),
        ]
    )
    model = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_data = (X_test, y_test)
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, y_prob)),
        "Brier Score": float(brier_score_loss(y_test, y_prob)),
    }


def get_feature_rankings(df):
    X = df[FEATURE_COLUMNS]
    y = df["cardio"]
    corr = X.corrwith(y).abs().sort_values(ascending=False).head(7)
    mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=FEATURE_COLUMNS)
    mi = mi.sort_values(ascending=False).head(7)

    scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=FEATURE_COLUMNS)
    chi_values, p_values = chi2(scaled, y)
    chi = pd.DataFrame({"score": chi_values, "p_value": p_values}, index=FEATURE_COLUMNS)
    chi = chi.sort_values("score", ascending=False).head(7)

    return {
        "correlation": [{"feature": k, "score": float(v)} for k, v in corr.items()],
        "mutual_information": [{"feature": k, "score": float(v)} for k, v in mi.items()],
        "chi_square": [
            {"feature": k, "score": float(row["score"])}
            for k, row in chi.iterrows()
        ],
    }


def ensure_initialized():
    global training_df
    if model is None:
        df, source = load_dataset()
        training_df = preprocess_data(df)
        app_state["metrics"] = train_model(training_df)
        app_state["rankings"] = get_feature_rankings(training_df)
        app_state["dataset"] = {
            "source": source,
            "records": int(training_df.shape[0]),
            "features": len(FEATURE_COLUMNS),
            "positive_rate": float(training_df["cardio"].mean()),
        }


def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/initialize", methods=["POST"])
def initialize():
    try:
        ensure_initialized()
        return jsonify(
            {
                "success": True,
                "metrics": app_state["metrics"],
                "rankings": app_state["rankings"],
                "features": FEATURE_DETAILS,
                "dataset": app_state["dataset"],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        ensure_initialized()
        payload = request.get_json(silent=True) or {}
        values = payload.get("values", {})
        missing = [feature for feature in FEATURE_COLUMNS if feature not in values]
        if missing:
            return jsonify({"error": f"Missing values: {', '.join(missing)}"}), 400

        row = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURE_COLUMNS}])
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(probability >= 0.5)
        risk_class = "High risk" if prediction else "Low risk"
        return jsonify(
            {
                "success": True,
                "risk_class": risk_class,
                "risk_percentage": probability * 100,
                "probabilities": {
                    "low_risk": (1 - probability) * 100,
                    "high_risk": probability * 100,
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    try:
        ensure_initialized()
        X_test, _ = test_data
        estimator = model.calibrated_classifiers_[0].estimator
        classifier = estimator.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            values = classifier.feature_importances_
        else:
            values = np.std(X_test, axis=0).to_numpy()
        importance = (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values})
            .sort_values("importance", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance, y="feature", x="importance", palette="crest", ax=ax)
        ax.set_title("Top Model Feature Importances")
        ax.set_xlabel("Relative importance")
        ax.set_ylabel("")
        return jsonify(
            {
                "success": True,
                "image": plot_to_base64(fig),
                "data": importance.to_dict("records"),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/confusion-matrix", methods=["GET"])
def confusion_matrix_route():
    try:
        ensure_initialized()
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=["Low", "High"],
            yticklabels=["Low", "High"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix on Holdout Data")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        return jsonify({"success": True, "image": plot_to_base64(fig)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
