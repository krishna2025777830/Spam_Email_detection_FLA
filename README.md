# Spam Email Detection (Finite Language Automata + ML)

## Quick Description

This project detects spam emails using Finite Language Automata (FLA) pattern matching combined with machine learning. It analyzes email text for 47 common spam phrases (like "free," "click," "win"), extracts features, and trains a classifier (Logistic Regression or Random Forest). Includes a Streamlit web UI for real-time detection, preprocessing pipeline, and unit tests. Built with Python, scikit-learn, pandas.

## About

This repository implements a hybrid spam detection system: a set of regex-based patterns (an automata-like rule set) extracts interpretable binary features from emails, and a classical supervised ML pipeline (Logistic Regression / Random Forest) learns to classify messages given those features. The approach emphasizes explainability (which patterns matched) alongside a statistical model for generalization.

**Note:** This project uses **FLA (Finite Language Automata) + Classical ML** (Logistic Regression, Random Forest), NOT Markov Chain or BERT models. The focus is on lightweight, interpretable spam detection suitable for resource-constrained environments.

## Problem

Unsolicited commercial and malicious emails (spam, phishing) are a longstanding problem. Many solutions rely on large transformer models or proprietary filtering systems. This project demonstrates a lightweight, open approach suitable for small-scale deployment or educational purposes:

- Detect suspicious tokens/phrases with clearly-defined patterns (e.g. `\bfree\b`, `\bclick here\b`).
- Convert these matches into binary features (pattern present / absent) plus simple numeric features (message length, token count).
- Train a robust classifier on labeled SMS/email data to decide spam vs ham.

This is useful where resource constraints or explainability are priorities.

## Methodology (detailed)

1. Data
   - The project expects an input CSV (example: `data/spam.csv`, common UCI/Kaggle SMS Spam dataset formatting). A preprocessing script produces `data/clean_spam.csv` with columns: `label, message, clean_text`.

2. Preprocessing
   - Lowercasing, URL replacement (`url` token), HTML tag removal, currency tokenization (`cur`), remove non-alphanumeric characters, collapse whitespace.
   - Light stopword removal (project-local stopword list) to reduce trivial tokens.
   - Label encoding: `ham -> 0`, `spam -> 1` (fallback rules for variants).

3. Feature Extraction (Automata / Patterns)
   - A pattern list (`app/automata.py`) defines regexes representing spammy phrases.
   - For each input message, the code tests each compiled regex and produces a feature `pat:<pattern_text>` with value 1 (match) or 0 (no match).
   - Additional features: `len_chars` (character length), `num_tokens` (word count).

4. Model Training
   - Feature matrix is prepared as a Pandas DataFrame.
   - Two classifiers are trained and compared on validation: Logistic Regression and Random Forest (see `app/train_model.py`). The pipeline uses StandardScaler + estimator.
   - The selected model (best validation F1) is saved with the pattern texts in a pickle (`data/spam_fla_model.pkl`).

5. Prediction / UI
   - `app/predict.py` loads the pickled model and pattern texts (compiling patterns if needed) and provides `real_time_detect(text, model, compiled_patterns)` returning prediction, probability, and the list of matched patterns.
   - `ui.py` is a Streamlit app for interactive testing.

## Repository structure

Key files and folders:

- `app/` — core logic: `automata.py`, `train_model.py`, `predict.py`.
- `data/` — input data, saved trained model, patterns JSON (not all large artifacts should be committed; see notes).
- `ui.py` — Streamlit UI.
- `requirements.txt` — pinned Python packages used in development.
- `tests/` — small pytest tests (e.g., `tests/test_predict.py`).

## Installation (Windows PowerShell)

Recommended: create a virtual environment and install pinned requirements. Run these from the inner project folder (where `requirements.txt` exists):

```powershell
# Change to project folder (adjust path to your environment)
Set-Location -LiteralPath "C:\Users\krish\OneDrive\Documents\B-Tech projects\spam_email (2)\spam_email"

# Create a venv (use a compatible Python version: 3.11 or 3.12 recommended)
py -3.12 -m venv .venv

# Allow activation in this PowerShell session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate
.\.venv\Scripts\Activate.ps1

# Upgrade pip/tools and install requirements
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Run tests
python -m pytest -q

# Run the Streamlit UI
python -m streamlit run ui.py
```

Notes:
- If you get C-extension/DLL import errors from numpy: recreate the venv with a Python version compatible with the wheels (3.11/3.12). Install the appropriate Microsoft Visual C++ Redistributable packages if needed.
- Large binary files (trained model `.pkl`) are better stored with Git LFS or published as a GitHub Release rather than committed to the main repo.

## Software used

- Python 3.11 / 3.12 (recommended)
- scikit-learn (classification, pipeline)
- pandas (data handling)
- numpy (numerical arrays)
- Streamlit (UI)
- pytest (tests)

Exact versions are pinned in `requirements.txt`.

## How to create a GitHub repository and upload (step-by-step)

Option A — Using the GitHub CLI (`gh`) — easiest and supports creating a remote repo from the terminal:

```powershell
# 1) Initialize git (if not already)
git init
git add --all
git commit -m "Initial commit: spam-email project"

# 2) Create a GitHub repo (interactive)
gh repo create my-username/spam_email --public --source=. --remote=origin

# 3) Push to GitHub
git branch -M main
git push -u origin main
```

Option B — Manual on GitHub website:

1. Create a new repository on https://github.com (click New repository). Don't initialize with README if you already have one locally.
2. Follow the provided instructions to add the remote URL: `git remote add origin https://github.com/<yourname>/<repo>.git` and push.

Large files note
- If you want to include `data/spam_fla_model.pkl`, consider using Git LFS:

```powershell
# Install git-lfs (one-time)
choco install git-lfs -y  # if using Chocolatey / or download installer from git-lfs.github.com
git lfs install
git lfs track "data/*.pkl"
git add .gitattributes
git add data/spam_fla_model.pkl
git commit -m "Add model with LFS"
git push origin main
```

## Suggested README contents for GitHub (this file)
- Keep this README.md and also add a short `USAGE.md` or `CONTRIBUTING.md` if you want collaborators.

## Security & Privacy
- Avoid committing sensitive data (private datasets, credentials, API keys). If your dataset is private, keep it out of Git and provide instructions for users to download or generate sample data.

## Next steps / Improvements

- Add model evaluation notebooks and plots (ROC, Precision/Recall, confusion matrix).
- Support more robust preprocessing (language detection, stemming, lemmatization).
- Add CI: a small GitHub Actions workflow that runs pytest on push.
- Add Dockerfile for reproducible runs.

---

If you want, I can:
- create the `README.md` (done), `.gitignore` (done), and optionally `.gitattributes` for LFS, and a GitHub Actions workflow.
- generate a `LICENSE` file (MIT/Apache/BSD) if you want to open-source it.
- run the exact `git` and `gh` commands here if you give me consent and your GitHub repo details or a personal access token (not recommended to paste tokens here). 

Tell me what next: create a GitHub repo for you (I'll provide commands), add a LICENSE, or create CI workflow.
