# ---- Streamlit App ----
import streamlit as st
import pandas as pd
import joblib
import shap
import os

# nltk preprocessing
import nltk
import re
import string
from nltk.corpus import stopwords

# 🚨 Download stopwords once if not already there
nltk_data_dir = os.path.expanduser('~') + '/nltk_data'
if not os.path.exists(nltk_data_dir + '/corpora/stopwords'):
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Set NLTK path manually to avoid error
nltk.data.path.append(nltk_data_dir)
stop_words = set(stopwords.words('english'))


# Load model and vectorizer
model = joblib.load("xgb_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
explainer = shap.TreeExplainer(model)

# Text preprocessing (same as in training)
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# SHAP interpretation function
def interpret_shap(shap_values, feature_names, sample):
    top_indices = abs(shap_values).argsort()[::-1][:5]
    output = []
    for i in top_indices:
        word = feature_names[i]
        contribution = shap_values[i]
        present = sample[i] > 0
        emoji = "🔴" if contribution > 0 and present else "🔵"
        direction = "positively" if contribution > 0 else "negatively"
        output.append(f"{emoji} **'{word}'** contributed {direction} ({contribution:.2f})")
    return "\n".join(output)

# Streamlit UI
st.title("🛡️ TrustIQ – Fake Review Detection")
user_input = st.text_area("Paste a product review to check if it's fake:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][pred]
        label = "❌ FAKE REVIEW" if pred == 1 else "✅ GENUINE REVIEW"

        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")

        # SHAP explanation
        vec_array = vec.toarray()
        shap_vals = explainer.shap_values(vec_array)
        explanation = interpret_shap(shap_vals[0], vectorizer.get_feature_names_out(), vec_array[0])
        st.markdown("### Explanation:")
        st.markdown(explanation)
