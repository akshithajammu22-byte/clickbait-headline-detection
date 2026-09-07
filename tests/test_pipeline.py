import os
import sys
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nlp_preprocessor import (
    clean_text_for_tfidf,
    extract_linguistic_features,
    extract_feature_vector,
    highlight_sensational_tokens
)
from predictor import ClickbaitPredictor

class TestClickbaitPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = ClickbaitPredictor()

    def test_clean_text_for_tfidf(self):
        sample = "10 SHOCKING Secrets Flight Attendants Don't Want You To Know!"
        cleaned = clean_text_for_tfidf(sample)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("10", cleaned)
        self.assertIn("shocking", cleaned)

    def test_linguistic_feature_extraction(self):
        cb_sample = "15 Mind-Blowing Facts You Won't Believe Actually Happened!"
        features = extract_linguistic_features(cb_sample)
        
        self.assertGreater(features["excl_count"], 0)
        self.assertEqual(features["starts_with_num"], 1)
        self.assertEqual(features["listicle_score"], 1)
        self.assertGreater(features["sensationalism_score"], 5.0)

    def test_feature_vector_dimensions(self):
        sample = "Federal Reserve Holds Benchmark Interest Rates Steady"
        vec = extract_feature_vector(sample)
        self.assertEqual(len(vec), 16)
        self.assertIsInstance(vec, np.ndarray)

    def test_sensational_token_highlighter(self):
        sample = "10 SHOCKING Secrets Will Leave You In Tears!"
        tokens = highlight_sensational_tokens(sample)
        self.assertTrue(any(t["word"] == "SHOCKING" and t["type"] in ["sensational-trigger", "caps-emphasis"] for t in tokens))
        self.assertTrue(any(t["type"] in ["sensational-trigger", "punctuation-spike"] for t in tokens))

    def test_predictor_clickbait_inference(self):
        cb_sample = "15 Mind-Blowing Facts You Won't Believe Actually Happened!"
        res = self.predictor.predict_single(cb_sample)
        self.assertTrue(res["is_clickbait"])
        self.assertGreater(res["clickbait_probability"], 60.0)
        self.assertIn("risk_level", res)
        self.assertIn("neutral_rewrite", res)

    def test_predictor_legitimate_inference(self):
        legit_sample = "As U.S. budget fight looms, Republicans flip their fiscal script"
        res = self.predictor.predict_single(legit_sample)
        self.assertFalse(res["is_clickbait"])
        self.assertLess(res["clickbait_probability"], 50.0)

    def test_predictor_batch_inference(self):
        batch = [
            "15 Mind-Blowing Facts You Won't Believe Actually Happened!",
            "As U.S. budget fight looms, Republicans flip their fiscal script"
        ]
        res = self.predictor.predict_batch(batch)
        self.assertEqual(res["summary"]["total_evaluated"], 2)
        self.assertEqual(len(res["results"]), 2)

if __name__ == "__main__":
    unittest.main()
