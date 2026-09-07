import os
import sys
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predictor import ClickbaitPredictor

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Lazy-loaded predictor singleton
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ClickbaitPredictor()
    return _predictor

SAMPLE_HEADLINES = [
    {
        "text": "15 Shocking Secrets Flight Attendants Never Tell Passengers!",
        "expected": "Clickbait",
        "category": "Curiosity Gap / Listicle"
    },
    {
        "text": "Federal Reserve Signals Possible Rate Reduction as Inflation Moderates",
        "expected": "Legitimate News",
        "category": "Economics & Policy"
    },
    {
        "text": "She Drank Celery Juice Every Day for a Week and What Happened to Her Skin Is Unreal",
        "expected": "Clickbait",
        "category": "Health & Lifestyle"
    },
    {
        "text": "NASA's James Webb Telescope Detects Organic Molecules in Distant Galaxy",
        "expected": "Legitimate News",
        "category": "Science & Astronomy"
    },
    {
        "text": "Why Everyone In Hollywood Is FURIOUS Over This Leaked Audio Clip",
        "expected": "Clickbait",
        "category": "Entertainment Sensationalism"
    },
    {
        "text": "United Nations Security Council Passes Resolution on Humanitarian Aid Access",
        "expected": "Legitimate News",
        "category": "World Politics"
    },
    {
        "text": "Stop Making This One Huge Mistake With Your Coffee Immediately",
        "expected": "Clickbait",
        "category": "Urgency & Warning"
    },
    {
        "text": "Global Semiconductor Market Forecasts 14 Percent Growth in Annual Revenue",
        "expected": "Legitimate News",
        "category": "Technology & Markets"
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    headline = data.get("headline", "").strip()
    if not headline:
        return jsonify({"error": "Please provide a non-empty 'headline' string in request body."}), 400
    
    predictor = get_predictor()
    result = predictor.predict_single(headline)
    return jsonify(result)

@app.route("/api/batch", methods=["POST"])
def batch_predict():
    predictor = get_predictor()
    
    # Handle CSV file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        
        try:
            content = file.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(content))
            
            # Look for headline or title column
            col_candidates = ["headline", "title", "text", "head", "news", "content"]
            target_col = None
            for col in df.columns:
                if col.lower() in col_candidates:
                    target_col = col
                    break
            if target_col is None:
                target_col = df.columns[0]
            
            headlines = df[target_col].dropna().astype(str).tolist()[:500]
            batch_res = predictor.predict_batch(headlines)
            return jsonify(batch_res)
        except Exception as e:
            return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 400

    # Handle JSON list of headlines
    data = request.get_json(silent=True) or {}
    headlines = data.get("headlines", [])
    if not isinstance(headlines, list) or len(headlines) == 0:
        return jsonify({"error": "Please provide a list of headlines in 'headlines' array or upload a CSV file."}), 400

    batch_res = predictor.predict_batch(headlines[:500])
    return jsonify(batch_res)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics not found. Training may still be in progress."}), 404

@app.route("/api/sample-headlines", methods=["GET"])
def sample_headlines():
    return jsonify(SAMPLE_HEADLINES)

@app.route("/api/dataset-stats", methods=["GET"])
def dataset_stats():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "clickbait_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        total = len(df)
        cb_count = int((df["label"] == 1).sum())
        legit_count = int((df["label"] == 0).sum())
        
        sample_rows = df.sample(min(30, total), random_state=42).to_dict(orient="records")
        
        return jsonify({
            "total_records": total,
            "clickbait_count": cb_count,
            "legitimate_count": legit_count,
            "categories": df["category"].value_counts().to_dict() if "category" in df.columns else {},
            "sample_rows": sample_rows
        })
    return jsonify({"error": "Dataset cache not yet available."}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Clickbait Detection Web App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
