import os
import sys
import argparse
import warnings
from importlib import import_module

from sklearn import metrics
import sklearn
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, chi2, RFE
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, CalibrationDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, brier_score_loss)

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
import joblib

def load_dataset():
    """Attempt to load a cardiovascular dataset from Kaggle or UCI."""
    df = None
    try:
        kaggle_module = import_module("kaggle.api.kaggle_api_extended")
        api = kaggle_module.KaggleApi()
        api.authenticate()
        dataset = 'sulianova/cardiovascular-disease-dataset'
        kaggle_file = 'cardio_train.csv'
        try:
            api.dataset_download_file(dataset, kaggle_file, path='.', force=True)
            df = pd.read_csv(kaggle_file, sep=';')
        except:
            api.dataset_download_files(dataset, path='.', unzip=True, force=True)
            if os.path.exists(kaggle_file):
                df = pd.read_csv(kaggle_file, sep=';')
    except Exception:
        pass

    
    if df is None:
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            df = pd.read_csv(url, header=None)
            
            df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                          "thalach","exang","oldpeak","slope","ca","thal","target"]
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("Dataset could not be loaded from Kaggle or UCI.")
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    
    if 'cardio' in df.columns or 'target' in df.columns:
        
        df = df.rename(columns={'target':'cardio'})
    else:
        
        df = df.rename(columns={df.columns[-1]:'cardio'})

    
    if 'age' in df.columns:
      
        if df['age'].mean() > 100:
            df['age_years'] = (df['age'] / 365).round().astype(int)
            df.drop('age', axis=1, inplace=True)
            df = df.rename(columns={'age_years':'age'})
    
    if 'gender' in df.columns and df['gender'].dtype == object:
        le = LabelEncoder()
        df['gender'] = le.fit_transform(df['gender'])

    if 'ap_hi' in df.columns:
        df = df[df['ap_hi'] > 0]
    if 'ap_lo' in df.columns:
        df = df[df['ap_lo'] > 0]

    df = df.replace("?", np.nan).dropna()
    
    # Convert all object-type columns to numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN after conversion
    df = df.dropna()
    
    if 'height' in df.columns and 'weight' in df.columns:
        df['BMI'] = df['weight'] / ((df['height']/100)**2)
    scaler = StandardScaler()
    cont_feats = [col for col in df.columns if col not in ['cardio','gender','cholesterol','gluc','smoke','alco','active']]
    for col in cont_feats:
        if df[col].dtype in [np.int64, np.float64]:
            df[col] = scaler.fit_transform(df[[col]])
    return df

def filter_ranking(df):
    """Compute filter-based feature rankings."""
    X = df.drop('cardio', axis=1)
    y = df['cardio'] 
    corr = {}
    for col in X.columns:
        if X[col].dtype != object:
            corr[col] = np.corrcoef(X[col], y)[0,1]
    corr_df = pd.DataFrame.from_dict(corr, orient='index', columns=['Correlation']).abs().sort_values(by='Correlation', ascending=False)
    mi = mutual_info_classif(X, y, random_state=0)
    mi_df = pd.DataFrame({'Feature':X.columns, 'MutualInfo':mi})
    mi_df = mi_df.sort_values(by='MutualInfo', ascending=False).set_index('Feature')
    chi_df = pd.DataFrame()
    try:
        X_cat = X.copy()
        for col in X_cat.columns:
            if X_cat[col].dtype == object or X_cat[col].dtype == bool:
                X_cat[col] = LabelEncoder().fit_transform(X_cat[col])
        chi2_vals, chi2_p = chi2(X_cat.fillna(0).astype(int), y)
        chi_df = pd.DataFrame({'Feature':X.columns, 'Chi2':chi2_vals, 'p_value':chi2_p})
        chi_df = chi_df.sort_values(by='Chi2', ascending=False).set_index('Feature')
    except Exception:
        pass
    return corr_df, mi_df, chi_df

def wrapper_selection(df):
    """Use RFE (wrapper) with gradient boosting to select features."""
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    if XGB_AVAILABLE:
        estimator = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    elif LGB_AVAILABLE:
        estimator = lgb.LGBMClassifier()
    else:
        estimator = GradientBoostingClassifier()
    n_select = max(1, X.shape[1] // 2)
    rfe = RFE(estimator, n_features_to_select=n_select, step=1)
    rfe.fit(X, y)
    ranking = pd.DataFrame({'Feature': X.columns, 'Rank': rfe.ranking_})
    ranking = ranking.sort_values(by='Rank')
    return ranking

def train_and_calibrate(df):
    """Train Gradient Boosting model and calibrate probabilities."""
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif LGB_AVAILABLE:
        model = lgb.LGBMClassifier()
    else:
        model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    # Use the correct approach for newer scikit-learn versions: fit calibrators on validation set
    platt = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    iso   = CalibratedClassifierCV(model, method='isotonic', cv=5)
    platt.fit(X_train, y_train)
    iso.fit(X_train, y_train)
    
    results = {}
    for name, clf in [('Uncalibrated', model), ('Platt', platt), ('Isotonic', iso)]:
        y_prob_full = clf.predict_proba(X_test)
        y_prob = y_prob_full[:,1] if y_prob_full.shape[1] == 2 else y_prob_full[:,1]
        y_pred = clf.predict(X_test)
        # Calculate Brier score (convert array to scalar for multiclass)
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.arange(y_prob_full.shape[1]))
        brier_val = float(np.mean((y_prob_full - y_test_bin)**2))
        results[name] = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'F1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'ROC AUC': float(roc_auc_score(y_test, y_prob_full, multi_class='ovo')),
            'Brier': brier_val
        }
    
    best_clf = iso
    cm = confusion_matrix(y_test, best_clf.predict(X_test))
    # For multiclass, ROC curve is not applicable, so we skip it
    fpr, tpr = None, None
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, best_clf.predict_proba(X_test)[:,1])
    return model, platt, iso, results, X_test, y_test, cm, (fpr, tpr)

def save_plots(results, best_clf, X_test, y_test, cm, roc):
    """Generate and save calibration and ROC plots.""" 
    # Only plot calibration curve for binary classification
    if len(np.unique(y_test)) == 2:
        fig, ax = plt.subplots(figsize=(6,6))
        CalibrationDisplay.from_estimator(best_clf, X_test, y_test, n_bins=10, name='Calibrated', ax=ax)
        ax.set_title("Calibration Curve (Isotonic)")
        fig.savefig("calibration_curve.png")
        plt.show()
    else:
        print("Calibration curve not available for multiclass classification.")
    
    # Only plot ROC curve if it's available (binary classification)
    if roc[0] is not None and roc[1] is not None:
        plt.figure()
        plt.plot(roc[0], roc[1], label=f'ROC (AUC = {results["Isotonic"]["ROC AUC"]:.2f})')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.show()
    else:
        print("ROC curve not available for multiclass classification.")

if __name__ == "__main__":
    print("=== Cardiovascular Risk Classification Script ===\n"
          "This script loads cardiovascular data, performs feature selection, trains a Gradient Boosting model,\n"
          "calibrates predicted probabilities (Platt and isotonic), and evaluates performance.\n")
    df = load_dataset()
    df = preprocess_data(df)
    print("Data loaded. Dataset shape:", df.shape)
    corr_df, mi_df, chi_df = filter_ranking(df)
    print("\n-- Filter-based Feature Rankings --")
    print("Correlation (abs) with target:\n", corr_df.head().to_string())
    print("\nMutual Information ranking:\n", mi_df.head().to_string())
    if not chi_df.empty:
        print("\nChi-square ranking:\n", chi_df.head().to_string())
    ranking = wrapper_selection(df)
    print("\n-- Wrapper-based Feature Ranking (RFE) --")
    print(ranking.to_string(index=False))
    model, platt, iso, results, X_test, y_test, cm, roc_data = train_and_calibrate(df)
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df = metrics_df.drop(columns=["ConfusionMatrix", "ROC"], errors="ignore")
    print("\n-- Evaluation Metrics --")
    print(metrics_df[['Accuracy','Precision','Recall','F1','ROC AUC','Brier']].round(3).to_string())
    print("\nConfusion Matrix (best model):\n", cm)
    save_plots(results, iso, X_test, y_test, cm, roc_data)
    print("\nCalibration and ROC curves saved as 'calibration_curve.png' and 'roc_curve.png'.")
    joblib.dump(model, "gb_model.joblib")
    joblib.dump(platt, "gb_platt_calibrator.joblib")
    joblib.dump(iso, "gb_iso_calibrator.joblib")
    print("Trained model and calibrators saved to disk.")
    print("\n=== Predict on a new patient ===")
    try:
        feat_order = list(df.drop('cardio', axis=1).columns)
        print("Enter patient data for features:", ", ".join(feat_order))
        raw = input("Enter values separated by commas (or 'exit'): ")
        if raw.strip().lower() != 'exit':
            values = raw.split(',')
            if len(values) != len(feat_order):
                print("Incorrect number of features provided.")
            else:
                vals = []
                for col, val in zip(feat_order, values):
                    if col == 'gender':
                        
                        val = val.strip()
                        vals.append(1 if val.lower() in ['m','male','1'] else 0)
                    else:
                        vals.append(float(val))
                prob = iso.predict_proba([vals])[0][1]
                pred = iso.predict([vals])[0]
                risk = "HIGH" if pred == 1 else "LOW"
                print(f"Predicted risk class: {risk} (Probability of disease: {prob*100:.1f}%)")
    except Exception as e:
        print("Prediction input aborted or invalid:", e)
