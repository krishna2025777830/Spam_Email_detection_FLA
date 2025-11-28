#!/usr/bin/env bash
# Train model and start Streamlit UI
python -m app.train_model --data-path data/clean_spam.csv --out data/spam_fla_model.pkl
streamlit run ui.py