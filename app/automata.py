"""
Automata and feature extraction utilities.
This file defines a set of regex patterns that represent the "spam language" and helper
functions to compile patterns, test matches (simulate automata acceptance), and extract features.
"""

import re
from typing import List, Tuple
import json
import os


# Default patterns (word-boundary aware). You can extend this list or load from JSON.
PATTERN_TEXTS = [
    r"\burl\b", r"\bcur\b",
    r"\bfree\b", r"\bwin(?:ner|s|ning)?\b", r"\bprize\b", r"\bclaim\b", r"\bcongrat",
    r"\burgent\b", r"\bclick\b", r"\boffer\b", r"\bbuy\b", r"\bcheap\b", r"\bdiscount\b",
    r"\blimited time\b", r"\bact now\b", r"\bcall now\b", r"\bcash\b", r"\bloan\b", r"\bcredit\b",
    r"\binvest(?:ment|or)?\b", r"\bearn\b", r"\bincome\b", r"\bguarantee\b", r"\brisk free\b", r"\blottery\b",
    r"\bwinner\b", r"\bcongratulations\b", r"\bsubscribe\b", r"\bunsubscribe\b", r"\bviagra\b", r"\bdeal\b",
    r"\binsurance\b", r"\bpassword\b", r"\baccount\b", r"\bverify\b", r"\bimportant\b",
    r"\bclick here\b", r"\bexclusive\b", r"\bapply now\b", r"\b100%\b", r"\bguaranteed\b", r"\bbonus\b",
    r"\bmillions?\b", r"\bdonation\b", r"\bsex\b", r"\bcheaply\b"
]

COMPILED_PATTERNS: List[Tuple[str, re.Pattern]] = [(p, re.compile(p)) for p in PATTERN_TEXTS]

def compile_patterns(pattern_texts: List[str]=None):
    #(Re)compile a list of regex patterns. Returns a list of tuples (pattern_text, compiled_pattern)."""
    pts = pattern_texts if pattern_texts is not None else PATTERN_TEXTS
    return [(p, re.compile(p)) for p in pts]

def automata_matches(text: str, compiled_patterns=COMPILED_PATTERNS):
    # Return list of pattern strings that match the input text (simulating finite automata acceptance)."""
    matches = []
    s = str(text).lower()
    for ptext, cre in compiled_patterns:
        if cre.search(s):
            matches.append(ptext)
    return matches

def extract_features_from_texts(texts, compiled_patterns=COMPILED_PATTERNS):
    rows = []
    for t in texts:
        row = {}
        ts = str(t).lower()
        for ptext, cre in compiled_patterns:
            row[f"pat:{ptext}"] = 1 if cre.search(ts) else 0
        row["len_chars"] = len(ts)
        row["num_tokens"] = len(ts.split())
        rows.append(row)
    return rows

# Helpers to persist/load patterns
def save_patterns(path: str, pattern_texts: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pattern_texts, f, indent=2)

def load_patterns(path: str):
    if not os.path.exists(path):
        return PATTERN_TEXTS
    import json
    with open(path, "r", encoding="utf-8") as f:
        pts = json.load(f)
    return pts