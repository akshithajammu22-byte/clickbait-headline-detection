import os
import re
import string
import joblib
import numpy as np
import scipy.sparse as sp

from nlp_preprocessor import (
    clean_text_for_tfidf,
    extract_feature_vector,
    extract_linguistic_features,
    highlight_sensational_tokens
)

class ClickbaitPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "clickbait_model.joblib")
        
        self.model_path = model_path
        self.artifacts = None
        self.model = None
        self.tfidf = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Loads serialized model pipeline from disk"""
        if not os.path.exists(self.model_path):
            print(f"[!] Model artifact not found at {self.model_path}. Training pipeline...")
            from train_models import train_and_evaluate_all
            train_and_evaluate_all()
        
        self.artifacts = joblib.load(self.model_path)
        self.model = self.artifacts["model"]
        self.tfidf = self.artifacts["tfidf"]
        self.scaler = self.artifacts["scaler"]
        self.best_model_name = self.artifacts.get("best_model_name", "Voting Ensemble")
        print(f"[*] Loaded inference engine with primary model: {self.best_model_name}")

    def generate_neutral_rewrite(self, headline):
        """
        AI heuristic rephraser to suggest an objective, non-sensational, journalistic rewrite.
        """
        text = str(headline).strip()
        
        # 1. Strip sensational intro phrases
        removals = [
            r"^(?:\d+\s+)?(?:mind-blowing|shocking|unbelievable|jaw-dropping|terrifying|hilarious)\s+facts?\s+(?:you\s+won't\s+believe\s+actually\s+happened|about)\s*",
            r"^you\s+won't\s+believe\s+(?:what|how|why)\s*",
            r"^this\s+(?:simple|one|bizarre)\s+(?:\d+[-\s]?minute\s+)?trick\s+will\s+",
            r"^what\s+happens\s+next\s+(?:will|left|is)\s*",
            r"^the\s+real\s+reason\s+why\s*",
            r"^top\s+\d+\s+secrets?\s+(?:doctors|experts|they)\s+don't\s+want\s+you\s+to\s+know\s+about\s*",
            r"^why\s+everyone\s+is\s+obsessed\s+with\s*",
            r"^stop\s+doing\s+this\s+(?:one\s+dangerous\s+thing\s+)?before\s*",
            r"^look\s+at\s+what\s+(?:this\s+celebrity\s+wore|happened)\s*",
            r"\s*-\s*jaw\s*dropping!?$",
            r"\s*\(#\d+\s+will\s+blow\s+your\s+mind\)!?$",
            r"\s*\[video\]!?$",
            r"\s*\[watch\]!?$",
        ]
        
        rewritten = text
        for pat in removals:
            rewritten = re.sub(pat, "", rewritten, flags=re.IGNORECASE)
            
        # 2. Fix casing (if mostly caps, capitalize properly)
        if sum(1 for c in rewritten if c.isupper()) / max(len(rewritten), 1) > 0.4:
            rewritten = rewritten.title()

        # 3. Clean trailing exclamation and redundant punctuation
        rewritten = re.sub(r"[!]+", ".", rewritten)
        rewritten = re.sub(r"\?+", "?", rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        # 4. Handle listicles (e.g. "15 Things About X" -> "Analysis and Overview of X")
        listicle_match = re.match(r"^(\d+)\s+(?:things|ways|reasons|tips|signs|secrets|photos)\s+(?:to|about|for|that)?\s*(.*)", rewritten, re.IGNORECASE)
        if listicle_match:
            rest = listicle_match.group(2).strip()
            if rest:
                rewritten = f"Overview of {rest.capitalize()}"

        if rewritten.lower() == text.lower() or len(rewritten) < 5:
            rewritten = f"Factual Summary: {text.strip(string.punctuation)} (Standard News Coverage)"

        return rewritten

    def predict_single(self, headline):
        """
        Runs comprehensive inference on a single headline:
        Returns probability, classification label, risk tier, linguistic breakdown,
        token highlights, and objective rewrite suggestion.
        """
        if not headline or not str(headline).strip():
            return {
                "error": "Headline text cannot be empty."
            }

        s_headline = str(headline).strip()
        cleaned = clean_text_for_tfidf(s_headline)
        
        # Extract features
        X_tfidf = self.tfidf.transform([cleaned])
        ling_vec = extract_feature_vector(s_headline).reshape(1, -1)
        ling_scaled = self.scaler.transform(ling_vec)
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(ling_scaled)]).tocsr()

        # Model Prediction
        try:
            proba = float(self.model.predict_proba(X_combined)[0][1]) * 100
        except Exception:
            pred = int(self.model.predict(X_combined)[0])
            proba = 85.0 if pred == 1 else 15.0

        is_clickbait = bool(proba >= 50.0)
        label = "Clickbait Headline" if is_clickbait else "Legitimate / Objective News"

        # Risk Classification
        if proba >= 80.0:
            risk_level = "Critical Clickbait"
            risk_color = "#ef4444" # red
            verdict = "Highly sensationalized with intense curiosity gaps, click triggers, and emotional baiting."
        elif proba >= 50.0:
            risk_level = "Moderate Clickbait"
            risk_color = "#f59e0b" # amber
            verdict = "Contains noticeable sensationalism or clickbait phrasing intended to drive traffic."
        elif proba >= 25.0:
            risk_level = "Mild / Standard Headline"
            risk_color = "#3b82f6" # blue
            verdict = "Generally informative with minor stylistic or engaging vocabulary."
        else:
            risk_level = "Legitimate Journalistic Headline"
            risk_color = "#10b981" # emerald
            verdict = "High journalistic integrity; direct, factual, and informative with neutral phrasing."

        linguistic_metrics = extract_linguistic_features(s_headline)
        tokens_highlighted = highlight_sensational_tokens(s_headline)
        neutral_rewrite = self.generate_neutral_rewrite(s_headline) if is_clickbait else s_headline

        return {
            "headline": s_headline,
            "prediction": label,
            "is_clickbait": is_clickbait,
            "clickbait_probability": round(proba, 1),
            "legitimate_probability": round(100.0 - proba, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "verdict": verdict,
            "model_used": self.best_model_name,
            "linguistic_breakdown": {
                "sensationalism_score": round(linguistic_metrics["sensationalism_score"], 1),
                "capitalization_pct": round(linguistic_metrics["cap_ratio"] * 100, 1),
                "all_caps_words": linguistic_metrics["all_caps_words"],
                "exclamation_marks": linguistic_metrics["excl_count"],
                "question_marks": linguistic_metrics["qmark_count"],
                "listicle_pattern": bool(linguistic_metrics["listicle_score"]),
                "clickbait_triggers_found": linguistic_metrics["trigger_count"],
                "word_count": linguistic_metrics["word_count"],
                "char_length": linguistic_metrics["char_len"],
                "average_word_len": round(linguistic_metrics["avg_word_len"], 1)
            },
            "highlighted_tokens": tokens_highlighted,
            "neutral_rewrite": neutral_rewrite
        }

    def predict_batch(self, headlines):
        """
        Runs batch inference across a list of headlines.
        Returns array of results with summary distribution.
        """
        results = []
        clickbait_count = 0
        legit_count = 0

        for h in headlines:
            if not str(h).strip():
                continue
            res = self.predict_single(h)
            if res.get("is_clickbait"):
                clickbait_count += 1
            else:
                legit_count += 1
            results.append(res)

        total = len(results)
        summary = {
            "total_evaluated": total,
            "clickbait_count": clickbait_count,
            "legitimate_count": legit_count,
            "clickbait_percentage": round((clickbait_count / total * 100), 1) if total > 0 else 0,
            "legitimate_percentage": round((legit_count / total * 100), 1) if total > 0 else 0
        }
        return {
            "summary": summary,
            "results": results
        }

if __name__ == "__main__":
    predictor = ClickbaitPredictor()
    test_1 = "10 Jaw-Dropping Secrets That Celebrities Don't Want You To Know!"
    test_2 = "Senate Passes Federal Infrastructure Spending Bill with Bipartisan Support"
    print("\n[TEST 1]:", predictor.predict_single(test_1))
    print("\n[TEST 2]:", predictor.predict_single(test_2))
