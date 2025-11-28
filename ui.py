# ui.py
import streamlit as st
from app.predict import load_model, real_time_detect

# Path to trained model
MODEL_PATH = "data/spam_fla_model.pkl"

# Load model + compiled regex patterns
model, compiled_patterns = load_model(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("Spam Email Detection (FLA + ML)")
st.write("Paste an email below and click **Detect** to see if it's Spam or Ham.")

# Input box
user_input = st.text_area("Email text:", height=200)

if st.button("Detect"):
    if user_input.strip():
        result = real_time_detect(user_input, model, compiled_patterns)

        # Prediction output
        if result["prediction"] == 1:
            st.error(f"Spam detected! (probability={result['prob_spam']:.2f})")
        else:
            st.success(f"Looks safe (Ham) (probability={result['prob_spam']:.2f})")

        # Matched automata rules
        if result["matched_rules"]:
            st.write("### Matched Rules")
            st.write(", ".join(result["matched_rules"]))
        else:
            st.write("### Matched Rules")
            st.write("_No suspicious patterns found._")
    else:
        st.warning("Please enter some text before clicking Detect.")
