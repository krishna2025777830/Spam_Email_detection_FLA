import os
import re
import csv

IN_DIR = os.path.dirname(__file__)
IN_FILE = "spam.csv"
OUT_DIR = IN_DIR
OUT_FILE = "clean_spam.csv"

URL_RE = re.compile(r'https?://\S+|www\.\S+')
CURRENCY_RE = re.compile(r'[$£€₹]')
HTML_TAG_RE = re.compile(r'<[^>]+>')
NON_ALPHA_RE = re.compile(r'[^a-z0-9\s]')
MULTISPACE_RE = re.compile(r'\s+')

STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","so","to","of","in","on","for",
    "is","am","are","was","were","be","been","being","at","by","with","as","from","that",
    "this","it","its","you","your","yours","me","my","we","our","ours","they","their",
    "them","he","she","his","her","him","do","does","did","doing","have","has","had"
}


def normalize_text(text: str) -> str:
    t = text.lower()
    t = URL_RE.sub(' url ', t)
    t = HTML_TAG_RE.sub(' ', t)
    t = CURRENCY_RE.sub(' cur ', t)
    t = NON_ALPHA_RE.sub(' ', t)
    t = MULTISPACE_RE.sub(' ', t).strip()
    return t


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def encode_label(lbl: str) -> int:
    v = str(lbl).strip().lower()
    if v == "ham":
        return 0
    if v == "spam":
        return 1
    return 1 if v in ("junk","unwanted","phish","spammy") else 0


def main():
    in_path = os.path.join(IN_DIR, IN_FILE)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found at {in_path}")

    # Read CSV
    with open(in_path, newline='', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError('Input CSV is empty')

    # Determine columns
    cols = rows[0].keys()
    label_col = None
    text_col = None
    if 'v1' in cols and 'v2' in cols:
        label_col = 'v1'
        text_col = 'v2'
    else:
        for c in cols:
            lc = c.lower()
            if label_col is None and lc in ("label", "class", "target", "category"):
                label_col = c
            if text_col is None and lc in ("message", "text", "sms"):
                text_col = c
    if label_col is None or text_col is None:
        raise ValueError('Could not find label/text columns in CSV')

    out_rows = []
    for r in rows:
        raw_label = r.get(label_col, '')
        message = r.get(text_col, '')
        if raw_label is None or message is None:
            continue
        raw_label = raw_label.strip()
        message = message.strip()
        if raw_label == '' or message == '':
            continue
        label = encode_label(raw_label)
        clean_text = normalize_text(message)
        clean_text = remove_stopwords(clean_text)
        out_rows.append({'label': label, 'message': message, 'clean_text': clean_text})

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    # Write output CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['label','message','clean_text'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f'Preprocessing (no pandas) complete. Saved cleaned data to: {out_path}')
    print('Preview:')
    for r in out_rows[:10]:
        print(r)

if __name__ == '__main__':
    main()
