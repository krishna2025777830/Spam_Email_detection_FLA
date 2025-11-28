#preprocess_spam.py
import os
import re
import pandas as pd

IN_DIR = "data"
IN_FILE = "spam.csv" 
OUT_DIR = "data"
OUT_FILE = "clean_spam.csv"

os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset(path):
    """
    Loads spam.csv and keeps only label/message.
    Handles common extra unnamed columns in Kaggle/UTM versions.
    """
    df = pd.read_csv(path, encoding="latin-1")
    # Some versions have columns: v1, v2, Unnamed: 2,3,4. Keep only v1,v2.
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label_raw", "message"]
    else:
        # Fallback: try common names
        label_col = None
        text_col = None
        for c in df.columns:
            lc = c.lower()
            if label_col is None and lc in ("label", "class", "target", "category"):
                label_col = c
            if text_col is None and lc in ("message", "text", "sms"):
                text_col = c
        if label_col is None or text_col is None:
            raise ValueError(
                "Could not find expected columns. Need (v1,v2) or (label/message)."
            )
        df = df[[label_col, text_col]].copy()
        df.columns = ["label_raw", "message"]

    # Drop empty rows
    df = df.dropna(subset=["label_raw", "message"]).reset_index(drop=True)
    return df

URL_RE = re.compile(r'https?://\S+|www\.\S+')
CURRENCY_RE = re.compile(r'[$£€₹]')
HTML_TAG_RE = re.compile(r'<[^>]+>')
NON_ALPHA_RE = re.compile(r'[^a-z0-9\s]')
MULTISPACE_RE = re.compile(r'\s+')

def normalize_text(text: str) -> str:
    """Lowercase, remove URLs, HTML, currency -> CUR, keep alphanum+space, collapse spaces."""
    t = text.lower()
    t = URL_RE.sub(' url ', t)
    t = HTML_TAG_RE.sub(' ', t)
    t = CURRENCY_RE.sub(' cur ', t)
    t = NON_ALPHA_RE.sub(' ', t)
    t = MULTISPACE_RE.sub(' ', t).strip()
    return t

STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","so","to","of","in","on","for",
    "is","am","are","was","were","be","been","being","at","by","with","as","from","that",
    "this","it","its","you","your","yours","me","my","we","our","ours","they","their",
    "them","he","she","his","her","him","do","does","did","doing","have","has","had"
}

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def encode_label(lbl: str) -> int:
    """
    Map ham->0, spam->1; otherwise try common variants.
    """
    v = str(lbl).strip().lower()
    if v == "ham":
        return 0
    if v == "spam":
        return 1
    # fallback: treat everything else as ham unless clearly spam-like
    return 1 if v in ("junk","unwanted","phish","spammy") else 0

def main():
    in_path = os.path.join(IN_DIR, IN_FILE)
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Input file not found at {in_path}. "
            "Place your spam.csv in the 'data' folder or adjust IN_DIR/IN_FILE."
        )

    df = load_dataset(in_path)

    # Encode label
    df["label"] = df["label_raw"].apply(encode_label)

    # Normalize text
    df["clean_text"] = df["message"].astype(str).apply(normalize_text)
    
    # Remove stopwords
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)

    # Keep final columns
    out_df = df[["label", "message", "clean_text"]].copy()

    # Save
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Preprocessing complete. Saved cleaned data to: {out_path}")
    print("Preview:")
    print(out_df.head(10))

if __name__ == "__main__":
    main()
