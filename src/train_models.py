import os
import sys
import json
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

from dataset_loader import load_and_preprocess_dataset
from nlp_preprocessor import clean_text_for_tfidf, extract_feature_vector

def train_and_evaluate_all():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("[*] CLICKBAIT HEADLINE DETECTION: MULTI-MODEL ML PIPELINE TRAINING", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Dataset
    df = load_and_preprocess_dataset(sample_size_per_class=6000, random_state=42)
    headlines = df["headline"].tolist()
    labels = df["label"].values

    print(f"[*] Preprocessing {len(headlines)} headlines with NLP Pipeline...", flush=True)
    cleaned_texts = [clean_text_for_tfidf(h) for h in headlines]
    
    # 2. Linguistic Feature Extraction
    print("[*] Extracting handcrafted linguistic & sensationalism indicators...", flush=True)
    ling_features = np.array([extract_feature_vector(h) for h in headlines], dtype=np.float32)

    # 3. Train-Test Split (80% Train, 20% Test)
    indices = np.arange(len(headlines))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train_text = [cleaned_texts[i] for i in train_idx]
    X_test_text = [cleaned_texts[i] for i in test_idx]
    X_train_ling = ling_features[train_idx]
    X_test_ling = ling_features[test_idx]

    # 4. Feature Transformations
    print("[*] Vectorizing with TF-IDF (Unigrams + Bigrams, Sublinear Scaling)...", flush=True)
    tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    print("[*] Normalizing linguistic feature matrices...", flush=True)
    scaler = StandardScaler()
    X_train_ling_scaled = scaler.fit_transform(X_train_ling)
    X_test_ling_scaled = scaler.transform(X_test_ling)

    # Combined Sparse Feature Matrix (TF-IDF + Linguistic Features)
    X_train_combined = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_ling_scaled)]).tocsr()
    X_test_combined = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_ling_scaled)]).tocsr()

    print(f"    - Training Shape: {X_train_combined.shape}", flush=True)
    print(f"    - Testing Shape:  {X_test_combined.shape}", flush=True)

    # 5. Initialize Candidate Classifiers
    print("\n[*] Initializing 5 Machine Learning Architectures...", flush=True)
    
    # Model 1: Logistic Regression
    clf_lr = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
    
    # Model 2: Calibrated Linear Support Vector Machine
    base_svc = LinearSVC(max_iter=2500, C=1.0, random_state=42)
    clf_svm = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    
    # Model 3: Multinomial Naive Bayes (trained on TF-IDF representation)
    clf_nb = MultinomialNB(alpha=0.3)
    
    # Model 4: Random Forest Classifier
    clf_rf = RandomForestClassifier(n_estimators=80, max_depth=25, random_state=42, n_jobs=1)
    
    # Model 5: Voting Ensemble (Soft Voting combining LR + Calibrated SVM + RF)
    clf_ensemble = VotingClassifier(
        estimators=[
            ("lr", clf_lr),
            ("svm", clf_svm),
            ("rf", clf_rf)
        ],
        voting="soft",
        n_jobs=1
    )

    models = {
        "Logistic Regression": (clf_lr, X_train_combined, X_test_combined),
        "Support Vector Machine (Calibrated)": (clf_svm, X_train_combined, X_test_combined),
        "Multinomial Naive Bayes": (clf_nb, X_train_tfidf, X_test_tfidf),
        "Random Forest Classifier": (clf_rf, X_train_combined, X_test_combined),
        "Voting Ensemble (LR + SVM + RF)": (clf_ensemble, X_train_combined, X_test_combined)
    }

    metrics_report = {}
    fitted_models = {}

    print("\n" + "=" * 85)
    print(f"{'MODEL NAME':<35} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'ROC-AUC':<10}")
    print("=" * 85)

    for name, (model, X_tr, X_te) in models.items():
        t0 = time.time()
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba) * 100
        except Exception:
            roc_auc = 0.0

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        cm = confusion_matrix(y_test, y_pred).tolist()

        train_duration = round(time.time() - t0, 2)
        fitted_models[name] = model

        metrics_report[name] = {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "roc_auc": round(roc_auc, 2),
            "confusion_matrix": cm,
            "training_time_sec": train_duration
        }

        print(f"{name:<35} | {acc:>9.2f}% | {prec:>9.2f}% | {rec:>9.2f}% | {f1:>9.2f}% | {roc_auc:>9.2f}%")

    print("=" * 85)

    # 6. Extract Top Clickbait & Journalistic Predictor Keywords (from Logistic Regression weights)
    vocab = {v: k for k, v in tfidf.vocabulary_.items()}
    lr_coefs = clf_lr.coef_[0][:len(vocab)]
    top_clickbait_idx = np.argsort(lr_coefs)[-30:][::-1]
    top_legitimate_idx = np.argsort(lr_coefs)[:30]

    top_clickbait_keywords = [
        {"word": vocab.get(idx, f"feature_{idx}"), "weight": round(float(lr_coefs[idx]), 3)}
        for idx in top_clickbait_idx
    ]
    top_legitimate_keywords = [
        {"word": vocab.get(idx, f"feature_{idx}"), "weight": round(float(abs(lr_coefs[idx])), 3)}
        for idx in top_legitimate_idx
    ]

    # Best Model Selection
    best_model_name = max(metrics_report, key=lambda k: metrics_report[k]["f1_score"])
    print(f"\n[BEST MODEL]: {best_model_name} (F1: {metrics_report[best_model_name]['f1_score']}%)", flush=True)

    # 7. Save Model Artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    artifacts = {
        "best_model_name": best_model_name,
        "model": fitted_models[best_model_name],
        "clf_lr": fitted_models["Logistic Regression"],
        "clf_svm": fitted_models["Support Vector Machine (Calibrated)"],
        "clf_ensemble": fitted_models["Voting Ensemble (LR + SVM + RF)"],
        "tfidf": tfidf,
        "scaler": scaler,
    }
    model_path = os.path.join(models_dir, "clickbait_model.joblib")
    joblib.dump(artifacts, model_path, compress=3)
    print(f"[*] Exported serialized ML pipeline to {model_path}", flush=True)

    # 8. Save Metrics JSON for Dashboard
    dataset_summary = {
        "total_samples": int(len(df)),
        "clickbait_samples": int((df["label"] == 1).sum()),
        "legitimate_samples": int((df["label"] == 0).sum()),
        "test_samples": int(len(y_test)),
        "features_count": int(X_train_combined.shape[1]),
        "tfidf_vocab_size": int(len(tfidf.vocabulary_)),
        "linguistic_features_count": int(X_train_ling.shape[1]),
        "total_pipeline_time": round(time.time() - start_time, 2)
    }

    full_metrics = {
        "dataset_summary": dataset_summary,
        "models": metrics_report,
        "best_model": best_model_name,
        "top_clickbait_keywords": top_clickbait_keywords,
        "top_legitimate_keywords": top_legitimate_keywords
    }

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[*] Exported detailed evaluation metrics to {metrics_path}", flush=True)

    print("\n[+] Training and Benchmarking Complete!", flush=True)
    return full_metrics

if __name__ == "__main__":
    train_and_evaluate_all()

