import streamlit as st
import pandas as pd
import joblib
import shap
import os

# === Page Config ===
st.set_page_config(page_title="TrustIQ - Fake Review Detector", page_icon="🛡️", layout="centered")

# === Load model and vectorizer ===
model = joblib.load("xgb_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
explainer = shap.TreeExplainer(model)

# === NLP preprocessing ===
import nltk
import re
import string
from nltk.corpus import stopwords

nltk_data_dir = os.path.expanduser('~') + '/nltk_data'
if not os.path.exists(nltk_data_dir + '/corpora/stopwords'):
    nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# === SHAP Interpretation ===
def interpret_shap_fixed(shap_values, feature_names, sample_vector, top_n=5):
    present_indices = [i for i, val in enumerate(sample_vector) if val > 0]
    contributions = [(feature_names[i], shap_values[i]) for i in present_indices]
    sorted_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]

    output = []
    for word, contrib in sorted_features:
        emoji = "🔴" if contrib > 0 else "🔹"
        direction = "positively" if contrib > 0 else "negatively"
        output.append(f"- {emoji} **'{word}'** contributed {direction} ({contrib:.2f})")
    return "\n".join(output)

# === UI ===
st.title("🛡️ TrustIQ: Fake Review Detector")
st.markdown("Check if a product review is genuine or fake using AI and NLP.\nPaste your review below and hit Analyze.")

with st.expander("ℹ️ How does TrustIQ work?"):
    st.markdown("""
    **TrustIQ** uses Natural Language Processing (NLP) and Machine Learning to classify product reviews as *Fake* or *Genuine*.

    - It cleans your review (removing stopwords, punctuation, etc.)
    - Converts it into numeric features using **TF-IDF vectorization**
    - Feeds it into a pre-trained **XGBoost classifier**
    - Interprets the prediction using **SHAP** to highlight important words

    This makes predictions **fast**, **transparent**, and **explainable**.
    """)

user_input = st.text_area("Paste a product review to check if it's fake:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        vec_array = vec.toarray()

        # Predict
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][pred]
        label = "❌ FAKE REVIEW" if pred == 1 else "✅ GENUINE REVIEW"

        # SHAP Explainability
        shap_vals = explainer.shap_values(vec_array)
        explanation = interpret_shap_fixed(shap_vals[0], vectorizer.get_feature_names_out(), vec_array[0])

        # Stylized badge output
        if pred == 1:
            st.markdown(f"<h3 style='color:red'>❌ FAKE REVIEW – {proba*100:.2f}% confident</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:green'>✅ GENUINE REVIEW – {proba*100:.2f}% confident</h3>", unsafe_allow_html=True)

        # SHAP explanation section
        with st.expander("🔍 See why this review was classified this way"):
            st.markdown(explanation)

# === Footer ===
st.markdown("---")
st.markdown("Built with ❤️ by **Bakhshish Sethi**")


# === Footer ===
st.markdown("---")
st.markdown("Built with ❤️ by **Bakhshish Sethi**")

