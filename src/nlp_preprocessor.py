import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure essential corpora are available
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()

# Strong Clickbait Trigger Keywords and Patterns
CLICKBAIT_TRIGGERS = [
    r"\byou won't believe\b",
    r"\bwhat happens next\b",
    r"\bthis is why\b",
    r"\bwill blow your mind\b",
    r"\bshocking\b",
    r"\bunbelievable\b",
    r"\breason why\b",
    r"\bsee what happened\b",
    r"\bcan you believe\b",
    r"\bsecret\b",
    r"\bsecrets\b",
    r"\bjaw[-\s]?dropping\b",
    r"\bmind[-\s]?blowing\b",
    r"\bbreaks (?:the )?internet\b",
    r"\bbreaks silence\b",
    r"\bmeltdown\b",
    r"\bdestroy(?:s|ed)\b",
    r"\bslam(?:s|med)\b",
    r"\bembarrassing\b",
    r"\brustles?\b",
    r"\btrapped\b",
    r"\bterrifying\b",
    r"\binsane\b",
    r"\bexposed\b",
    r"\bmust see\b",
    r"\bmust watch\b",
    r"\bnever seen before\b",
    r"\bthe truth about\b",
    r"\bhilarious\b",
    r"\bepic\b",
    r"\bgenius\b",
    r"\brate this\b",
    r"\bwon't tell you\b",
    r"\bstop doing\b",
    r"\btell all\b",
    r"\bleaves everyone\b",
    r"\bgoes viral\b",
    r"\bleft in tears\b",
    r"\btop \d+\b",
    r"^\d+\s+\w+",  # Listicle style (e.g. 15 Things...)
]

QUESTION_WORDS = {"why", "what", "how", "who", "when", "where", "which", "could", "would", "is", "are", "can"}

def clean_text_for_tfidf(text):
    """
    Cleans raw headline text for TF-IDF vectorization:
    - Lowercases text
    - Removes URLs, HTML tags, special symbols
    - Lemmatizes and filters stopwords
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\(reuters\)\s*-\s*", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(cleaned_tokens) if cleaned_tokens else "empty"

def extract_linguistic_features(text):
    """
    Extracts numerical NLP / linguistic features characterizing sensationalism and clickbait:
    1. Capitalization Ratio (Upper case char count / total chars)
    2. All-Caps Words Count (e.g. SHOCKING, MUST SEE)
    3. Exclamation Marks Count (!)
    4. Question Marks Count (?)
    5. Punctuation Intensity (total ! ? ... " ')
    6. Starts With Number / Listicle Indicator (0 or 1)
    7. Number of Digits in Headline
    8. Starts With Question Word (0 or 1)
    9. Clickbait Trigger Matches Count
    10. Word Count
    11. Character Length
    12. Average Word Length
    13. Punctuation to Character Ratio
    14. Sensationalism Score (combined heuristic 0-100)
    """
    s_text = str(text).strip()
    char_len = max(len(s_text), 1)
    words = s_text.split()
    word_count = max(len(words), 1)

    # Capitalization metrics
    upper_chars = sum(1 for c in s_text if c.isupper())
    cap_ratio = upper_chars / char_len
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha())

    # Punctuation metrics
    excl_count = s_text.count("!")
    qmark_count = s_text.count("?")
    ellipsis_count = 1 if "..." in s_text or "…" in s_text else 0
    quotes_count = s_text.count('"') + s_text.count("'")
    total_punct = sum(1 for c in s_text if c in string.punctuation)
    punct_ratio = total_punct / char_len

    # Listicle & Number detection
    starts_with_num = 1 if re.match(r"^\d+", s_text) else 0
    digit_count = sum(1 for c in s_text if c.isdigit())
    listicle_score = 1 if (starts_with_num or bool(re.search(r"\b\d+\s+(?:things|ways|reasons|photos|secrets|tips|tricks|facts|signs)\b", s_text, re.I))) else 0

    # Question Word Structure
    first_word = words[0].lower().strip(string.punctuation) if words else ""
    starts_with_q = 1 if first_word in QUESTION_WORDS else 0

    # Clickbait Trigger Regex Counts
    text_lower = s_text.lower()
    trigger_count = sum(1 for pattern in CLICKBAIT_TRIGGERS if re.search(pattern, text_lower))

    # Average word length
    avg_word_len = sum(len(w) for w in words) / word_count

    # Sensationalism heuristic composite score (0 to 10)
    sensational_points = (
        (trigger_count * 2.5) +
        (3.0 if cap_ratio > 0.3 else (1.5 if cap_ratio > 0.15 else 0)) +
        (2.0 if excl_count > 0 else 0) +
        (1.5 if qmark_count > 0 else 0) +
        (2.0 if listicle_score > 0 else 0) +
        (1.5 if all_caps_words >= 2 else 0)
    )
    sensationalism_score = min(sensational_points, 10.0)

    feature_dict = {
        "cap_ratio": cap_ratio,
        "all_caps_words": all_caps_words,
        "excl_count": excl_count,
        "qmark_count": qmark_count,
        "ellipsis_count": ellipsis_count,
        "quotes_count": quotes_count,
        "punct_ratio": punct_ratio,
        "starts_with_num": starts_with_num,
        "digit_count": digit_count,
        "listicle_score": listicle_score,
        "starts_with_q": starts_with_q,
        "trigger_count": trigger_count,
        "word_count": word_count,
        "char_len": char_len,
        "avg_word_len": avg_word_len,
        "sensationalism_score": sensationalism_score,
    }
    return feature_dict

def extract_feature_vector(text):
    """Returns numerical feature vector array in fixed order"""
    d = extract_linguistic_features(text)
    return np.array([
        d["cap_ratio"],
        d["all_caps_words"],
        d["excl_count"],
        d["qmark_count"],
        d["ellipsis_count"],
        d["quotes_count"],
        d["punct_ratio"],
        d["starts_with_num"],
        d["digit_count"],
        d["listicle_score"],
        d["starts_with_q"],
        d["trigger_count"],
        d["word_count"],
        d["char_len"],
        d["avg_word_len"],
        d["sensationalism_score"]
    ], dtype=np.float32)

def highlight_sensational_tokens(text):
    """
    Identifies specific sensational, clickbait, and hyperbolic words in the headline
    and returns token metadata with classification highlights.
    """
    s_text = str(text)
    tokens = s_text.split()
    highlighted = []
    
    sensational_words = {
        "shocking", "unbelievable", "secret", "secrets", "mind-blowing", "mindblowing",
        "jaw-dropping", "insane", "exposed", "truth", "viral", "miracle", "epic",
        "hilarious", "terrifying", "genius", "ruining", "warning", "bizarre",
        "tears", "shocks", "meltdown", "slams", "destroyed", "horrifying", "unreal"
    }

    for token in tokens:
        clean_w = token.lower().strip(string.punctuation)
        is_trigger = clean_w in sensational_words
        is_caps = token.isupper() and len(token) > 1 and token.isalpha()
        is_punct = any(p in token for p in ["!", "?", "..."])
        
        tag = "normal"
        if is_trigger:
            tag = "sensational-trigger"
        elif is_caps:
            tag = "caps-emphasis"
        elif is_punct:
            tag = "punctuation-spike"
            
        highlighted.append({
            "word": token,
            "type": tag
        })
    return highlighted

if __name__ == "__main__":
    test_headline = "10 SHOCKING Secrets Doctors Don't Want You To Know! (#4 Will Blow Your Mind)"
    print("Clean TF-IDF text:", clean_text_for_tfidf(test_headline))
    print("Linguistic Features:", extract_linguistic_features(test_headline))
    print("Highlighted Tokens:", highlight_sensational_tokens(test_headline))
