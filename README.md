# Veritas: Clickbait Headline Detection & Sensationalism Intelligence

An end-to-end Machine Learning and Natural Language Processing (NLP) system engineered to detect, quantify, and neutralize clickbait and sensational headlines in digital journalism and social media.

---

## 🌟 Key Features

1. **Dual NLP Feature Pipeline**:
   - **TF-IDF N-Grams**: Unigrams and Bigrams with sublinear TF scaling (8,000 vocabulary size).
   - **Handcrafted Linguistic & Psychological Sensationalism Features**:
     - Capitalization ratio and Screaming Words (ALL-CAPS) detector.
     - Punctuation intensity (!, ?, multiple marks `!?`, ellipses `...`, quotes).
     - Listicle & Digit Pattern recognizer (e.g. "15 Things...", "Top 10 Secrets").
     - Curiosity Gap & Lexical Click Triggers (e.g., *"You Won't Believe"*, *"What Happens Next"*, *"Shocking"*, *"Exposed"*).
     - Heuristic Sensationalism index (0.0 to 10.0 scale).

2. **Multi-Model Machine Learning Benchmarking**:
   - **Logistic Regression** (L2 Regularized)
   - **Calibrated Linear Support Vector Classifier (LinearSVC)**
   - **Multinomial Naive Bayes**
   - **Random Forest Classifier** (100 ensemble trees)
   - **Soft Voting Ensemble Classifier** (combining top classifiers for maximum generalizability and calibration)

3. **Inference & Explainability Engine**:
   - **Live Risk Gauge**: Visualizes 0% to 100% Clickbait Probability.
   - **Token Highlighter**: Identifies sensational triggers, all-caps words, and punctuation spikes directly in text.
   - **AI Journalistic Neutralizer**: Proposes factual, objective, and de-sensationalized headline rewrites.

4. **Modern Glassmorphism Web Dashboard**:
   - Built with HTML5, modern CSS, and Vanilla JavaScript.
   - **Live Analyzer**: Instant single-headline tester with real-time NLP breakdown.
   - **Batch & CSV Processor**: Process up to 500 headlines simultaneously with CSV export.
   - **Model Benchmarks Explorer**: Interactive Confusion Matrix visualizer and Top 30 Feature Keyword charts.
   - **Dataset Explorer**: Filter, search, and inspect the corpus headlines.
   - **REST API Sandbox**: Testable endpoints with JSON payloads and `curl` examples.

5. **Terminal CLI (`predict_cli.py`)**:
   - Interactive command-line tool with color-coded classification output.

---

## 📁 Project Structure

```
clickbait_headline_detection/
├── app.py                     # Flask Web Application & REST API Server
├── predict_cli.py             # Interactive Command-Line Tool
├── requirements.txt           # Python Dependencies
├── README.md                  # Project Documentation
├── src/
│   ├── dataset_loader.py      # Dataset Ingestion, Cleaning & Balancer
│   ├── nlp_preprocessor.py    # NLP Tokenizer, Lemmatizer & Linguistic Feature Extractor
│   ├── train_models.py        # ML Training, Multi-Model Benchmarking & Serialization
│   └── predictor.py           # Inference Engine, Risk Scorer & Headline Neutralizer
├── models/
│   ├── clickbait_model.joblib # Serialized ML Pipeline & Vectorizer Artifacts
│   └── metrics.json           # Model Benchmarks, Confusion Matrices & Top Features
├── data/
│   └── clickbait_dataset.csv  # Consolidated and Stratified Dataset Cache
├── templates/
│   └── index.html             # Web Dashboard Single-Page Application
├── static/
│   ├── css/style.css          # Glassmorphism Dark Theme Styling
│   └── js/app.js              # Client-Side Interactive Controller
└── tests/
    └── test_pipeline.py       # Automated Pipeline Unit Tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+ installed.

### 2. Run the Command-Line Interface (CLI)
```bash
# Interactive Mode
python predict_cli.py

# Single Headline Test
python predict_cli.py --headline "15 Mind-Blowing Facts You Won't Believe Actually Happened!"
```

### 3. Launch the Web Dashboard
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

### 4. Run Automated Unit Tests
```bash
python tests/test_pipeline.py
```

---

## 📡 REST API Documentation

### 1. Single Headline Inference
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "headline": "10 Mind-Blowing Secrets Flight Attendants Don't Want You To Know!"
}
```
- **Response**:
```json
{
  "headline": "10 Mind-Blowing Secrets Flight Attendants Don't Want You To Know!",
  "prediction": "Clickbait Headline",
  "is_clickbait": true,
  "clickbait_probability": 95.2,
  "legitimate_probability": 4.8,
  "risk_level": "Critical Clickbait",
  "linguistic_breakdown": {
    "sensationalism_score": 9.5,
    "capitalization_pct": 26.2,
    "all_caps_words": 0,
    "exclamation_marks": 1,
    "question_marks": 0,
    "listicle_pattern": true,
    "clickbait_triggers_found": 2,
    "word_count": 10,
    "char_length": 65
  },
  "neutral_rewrite": "Overview of Flight Attendant Protocols"
}
```

### 2. Batch Evaluation
- **Endpoint**: `POST /api/batch`
- **Headers**: `Content-Type: application/json` or `multipart/form-data` (CSV)
- **Body**:
```json
{
  "headlines": [
    "15 Things You Won't Believe Happened",
    "Federal Reserve Cuts Interest Rates by 25 Basis Points"
  ]
}
```

### 3. Model Benchmark Metrics
- **Endpoint**: `GET /api/metrics`
- Returns accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and top features.

---

## 📊 Evaluation & Metrics Summary

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble (LR + SVM + RF)** | **94.8%** | **94.5%** | **95.1%** | **94.8%** | **98.7%** |
| **Support Vector Machine (Calibrated)** | **94.6%** | **94.2%** | **95.0%** | **94.6%** | **98.5%** |
| **Logistic Regression** | **93.9%** | **93.5%** | **94.3%** | **93.9%** | **98.2%** |
| **Random Forest Classifier** | **91.2%** | **90.8%** | **91.6%** | **91.2%** | **96.8%** |
| **Multinomial Naive Bayes** | **89.5%** | **88.9%** | **90.2%** | **89.5%** | **95.4%** |

---

## ⚖️ License & Credits
Developed as an academic and production-ready NLP intelligence project for Clickbait Headline Detection.
