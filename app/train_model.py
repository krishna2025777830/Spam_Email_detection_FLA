#Training script. Loads the cleaned CSV, extracts automata features, trains classifiers,

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from app.automata import compile_patterns, extract_features_from_texts, save_patterns


def make_feature_dataframe(rows):
    # convert list of dicts to pandas DataFrame
    import pandas as pd
    return pd.DataFrame(rows)

def train_and_select(X_train, y_train, X_val, y_val):
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))


    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)


    pred_lr = pipe_lr.predict(X_val)
    pred_rf = pipe_rf.predict(X_val)


    f1_lr = f1_score(y_val, pred_lr, zero_division=0)
    f1_rf = f1_score(y_val, pred_rf, zero_division=0)


    if f1_rf >= f1_lr:
        return pipe_rf, 'RandomForest', f1_rf
    else:
        return pipe_lr, 'LogisticRegression', f1_lr

def main(args):
    df = pd.read_csv(args.data_path)
    # expected: columns 'clean_text' and 'label' (0/1). adapt if different.
    if 'clean_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Expected 'clean_text' and 'label' columns in the CSV.")

    texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].astype(int).values

    # compile patterns from default or provided json
    patterns = compile_patterns()

    rows = extract_features_from_texts(texts, compiled_patterns=patterns)
    X = make_feature_dataframe(rows)
    y = labels

    # train-val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model, model_name, sel_f1 = train_and_select(X_train, y_train, X_val, y_val)

    print(f"Selected model: {model_name} with validation F1: {sel_f1:.4f}")
    print("Classification report on validation set:")
    preds = model.predict(X_val)
    print(classification_report(y_val, preds, zero_division=0))

    # Save model and pattern texts
    out_obj = {
        'model': model,
        'patterns': [p for p, _ in [(pt, None) for pt in []]] # placeholder, fill below
    }
    # we want to save the textual patterns used (so loadable)
    out_obj['patterns'] = [p for p, _ in [(pt, None) for pt in []]]
    # Instead use the original pattern texts from automata module: import directly
    from app.automata import PATTERN_TEXTS
    out_obj['patterns'] = PATTERN_TEXTS

    with open(args.out, 'wb') as f:
        pickle.dump(out_obj, f)

    # also save patterns as json next to model
    import json, os
    ppath = os.path.splitext(args.out)[0] + '_patterns.json'
    with open(ppath, 'w', encoding='utf-8') as f:
        json.dump(PATTERN_TEXTS, f, indent=2)

    print(f"Saved model to: {args.out}")
    print(f"Saved patterns to: {ppath}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True, help='Path to clean_spam.csv')
    p.add_argument('--out', required=True, help='Output path for trained model (pickle)')
    args = p.parse_args()
    main(args)