#Load saved model and do real-time prediction

import pickle
from typing import Dict, Any, List, Tuple
import os
import re
from app.automata import compile_patterns, automata_matches, extract_features_from_texts
import pandas as pd


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    model = obj.get('model') if isinstance(obj, dict) else obj
    patterns = obj.get('patterns') if isinstance(obj, dict) else None
    # patterns may be stored as a list of strings, or as a list of (pattern_text, compiled_pattern) tuples.
    if patterns is None:
        compiled = compile_patterns()
    else:
        # If patterns is a list of strings, use compile_patterns to compile them.
        if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
            compiled = compile_patterns(patterns)
        else:
            # Ensure we return a list of (pattern_text, compiled_regex)
            compiled = []  # type: List[Tuple[str, re.Pattern]]
            for p in patterns:
                if isinstance(p, tuple) and len(p) >= 2 and hasattr(p[1], 'search'):
                    # Already compiled
                    compiled.append((p[0], p[1]))
                elif isinstance(p, str):
                    compiled.append((p, re.compile(p)))
                else:
                    # Fallback: coerce to string and compile
                    txt = str(p)
                    compiled.append((txt, re.compile(txt)))
    return model, compiled

def real_time_detect(text: str, model, compiled_patterns) -> Dict[str, Any]:
    t = str(text)
    matched = automata_matches(t, compiled_patterns=compiled_patterns)
    # build features
    rows = extract_features_from_texts([t], compiled_patterns=compiled_patterns)
    df = pd.DataFrame(rows)
    pred = int(model.predict(df)[0])
    prob = None
    try:
        prob = float(model.predict_proba(df)[:, 1][0])
    except Exception:
        prob = None
    return {
        'prediction': pred,
        'prob_spam': prob,
        'matched_rules': matched,
        'input': text
    }