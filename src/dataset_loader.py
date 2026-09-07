import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATHS = [
    os.path.join(r"C:\Users\hemas\OneDrive\Desktop\demoapp", "Fake.csv"),
    os.path.join(r"C:\Users\hemas\OneDrive\Desktop\demoapp", "True.csv"),
    os.path.join(os.path.dirname(__file__), "..", "data", "Fake.csv"),
    os.path.join(os.path.dirname(__file__), "..", "data", "True.csv"),
    "Fake.csv",
    "True.csv"
]

CURATED_CLICKBAIT_SAMPLES = [
    ("15 Mind-Blowing Facts You Won't Believe Actually Happened!", 1, "curated_clickbait"),
    ("This Simple 2-Minute Trick Will Change Your Life Forever", 1, "curated_clickbait"),
    ("She Opened The Mystery Box And What Happened Next Left Everyone In Tears", 1, "curated_clickbait"),
    ("Top 10 Secrets Doctors Don't Want You To Know About Weight Loss", 1, "curated_clickbait"),
    ("You Will Never Eat Fast Food Again After Seeing THIS Shocking Video", 1, "curated_clickbait"),
    ("Why Millenials Are Completely Ruining The Diamond Industry", 1, "curated_clickbait"),
    ("25 Ridiculously Genius Products That Are Selling Out Fast", 1, "curated_clickbait"),
    ("Scientists Discovered Something Terrifying At The Bottom Of The Ocean", 1, "curated_clickbait"),
    ("Here Is The Real Reason Why Flight Attendants Never Drink Coffee On Planes", 1, "curated_clickbait"),
    ("12 Unreal Photos That Prove We Live In A Simulation", 1, "curated_clickbait"),
    ("Everyone Is Obsessed With This Incredible Hair Secret", 1, "curated_clickbait"),
    ("This Is What Happens To Your Brain When You Quit Sugar For 30 Days", 1, "curated_clickbait"),
    ("Unbelievable! Man Finds Rare Treasure In His Own Backyard", 1, "curated_clickbait"),
    ("Can You Pass This Impossible 90s Trivia Quiz Without Failing?", 1, "curated_clickbait"),
    ("7 Warning Signs Your Body Is Begging For More Sleep", 1, "curated_clickbait"),
    ("The Bizarre Reason Why Millionaires Wake Up At 4 AM Every Morning", 1, "curated_clickbait"),
    ("Look At What This Celebrity Wore On The Red Carpet - Jaw Dropping!", 1, "curated_clickbait"),
    ("Stop Doing This One Dangerous Thing Before Going To Sleep", 1, "curated_clickbait"),
    ("Federal Reserve Holds Interest Rates Steady At 5.25 Percent", 0, "curated_news"),
    ("NASA Launches Artemis Spacecraft On Lunar Exploration Mission", 0, "curated_news"),
    ("United Nations Summit Concludes With Global Climate Accord", 0, "curated_news"),
    ("European Central Bank Issues Quarterly Economic Forecast", 0, "curated_news"),
    ("Tech Giant Unveils Quantum Computing Chip With 1000 Qubits", 0, "curated_news"),
    ("World Health Organization Reports Drop In Seasonal Influenza Cases", 0, "curated_news"),
    ("Japan Approves Renewable Energy Infrastructure Funding Package", 0, "curated_news"),
    ("Astronomers Discover Water Vapor Signatures In Exoplanet Atmosphere", 0, "curated_news"),
    ("Stock Market Indexes Close Mixed Following Labor Department Report", 0, "curated_news"),
    ("High Court Delivers Ruling In Interstate Commerce Dispute", 0, "curated_news"),
    ("New Archaeological Excavation Reveals Roman Settlement Remains", 0, "curated_news"),
    ("Treasury Department Releases Monthly Fiscal Position Statement", 0, "curated_news"),
]

def find_existing_dataset():
    fake_path, true_path = None, None
    for p in DATASET_PATHS:
        if "Fake.csv" in p and os.path.exists(p) and fake_path is None:
            fake_path = p
        if "True.csv" in p and os.path.exists(p) and true_path is None:
            true_path = p
    return fake_path, true_path

def load_and_preprocess_dataset(sample_size_per_class=12000, random_state=42):
    """
    Loads headlines from Fake.csv (labeled as Clickbait/Sensational = 1) 
    and True.csv (labeled as Non-Clickbait/Objective News = 0).
    Adds curated benchmark examples and returns a balanced, cleaned DataFrame.
    """
    fake_path, true_path = find_existing_dataset()
    frames = []

    if fake_path and true_path and os.path.exists(fake_path) and os.path.exists(true_path):
        print(f"[*] Found primary datasets:\n    - Fake/Clickbait: {fake_path}\n    - Real/Journalistic: {true_path}")
        
        # Read title column and subject column
        df_fake = pd.read_csv(fake_path, usecols=lambda c: c in ["title", "subject"])
        df_fake = df_fake.dropna(subset=["title"])
        df_fake["headline"] = df_fake["title"].astype(str).str.strip()
        df_fake["label"] = 1  # Clickbait / Sensational
        df_fake["category"] = df_fake.get("subject", "Sensational News")

        df_true = pd.read_csv(true_path, usecols=lambda c: c in ["title", "subject"])
        df_true = df_true.dropna(subset=["title"])
        df_true["headline"] = df_true["title"].astype(str).str.strip()
        df_true["label"] = 0  # Non-clickbait / Objective News
        df_true["category"] = df_true.get("subject", "Mainstream News")

        # Drop duplicates and blank titles
        df_fake = df_fake.drop_duplicates(subset=["headline"])
        df_true = df_true.drop_duplicates(subset=["headline"])
        df_fake = df_fake[df_fake["headline"].str.len() > 10]
        df_true = df_true[df_true["headline"].str.len() > 10]

        # Sample for balanced high-speed training if dataset is large
        if len(df_fake) > sample_size_per_class:
            df_fake = df_fake.sample(n=sample_size_per_class, random_state=random_state)
        if len(df_true) > sample_size_per_class:
            df_true = df_true.sample(n=sample_size_per_class, random_state=random_state)

        frames.extend([df_fake[["headline", "label", "category"]], df_true[["headline", "label", "category"]]])

    # Add curated clickbait benchmark samples
    df_curated = pd.DataFrame(CURATED_CLICKBAIT_SAMPLES, columns=["headline", "label", "category"])
    frames.append(df_curated)

    combined_df = pd.concat(frames, axis=0, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[*] Dataset successfully loaded:")
    print(f"    - Total Headlines: {len(combined_df)}")
    print(f"    - Clickbait (Class 1): {(combined_df['label'] == 1).sum()}")
    print(f"    - Non-Clickbait (Class 0): {(combined_df['label'] == 0).sum()}")

    # Save cached CSV for quick future access
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "clickbait_dataset.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"[*] Saved consolidated dataset to {out_path}")

    return combined_df

if __name__ == "__main__":
    df = load_and_preprocess_dataset()
    print(df.head())
