# 🛡️ TrustIQ — Fake Review Classifier with XGBoost & SHAP Explainability

https://trustiq-v2.streamlit.app/?

TrustIQ is a real-time, interactive app that detects fake product reviews using NLP and machine learning. Built with XGBoost and SHAP, it not only predicts whether a review is fake or genuine, but also explains *why* — giving users transparency and insight into the decision-making process.

---

## 🚀 Features

- 🔍 **Review classification**: Predicts whether a product review is fake or genuine
- 🧠 **NLP pipeline**: Cleans and vectorizes text using TF-IDF
- 🌲 **XGBoost model**: Trained with SMOTE to handle class imbalance
- 📊 **Explainability**: Uses SHAP to highlight which words influenced the prediction
- 💬 **Interactive UI**: Paste any review and get instant prediction, confidence score, and explanation
- ☁️ **Deployed on Streamlit Cloud**

---

## 🧰 Tech Stack

- Python 🐍  
- Scikit-learn  
- XGBoost  
- SHAP  
- Pandas, NumPy  
- Streamlit  
- NLTK (stopwords)

---

## 🛠️ How It Works

1. **Input**: User pastes a product review into the app
2. **Preprocessing**: The text is cleaned, tokenized, and vectorized with TF-IDF
3. **Prediction**: An XGBoost model classifies the review as fake or genuine
4. **Explainability**: SHAP highlights the top contributing words
5. **Output**: Displayed in a clean UI with confidence score and bullet-point explanation

---

