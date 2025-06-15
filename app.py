# app.py - Streamlit GUI for Fake News Detection
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = PassiveAggressiveClassifier(max_iter=50)
    df = pd.read_csv("fake_or_real_news.csv")
    df.dropna(inplace=True)
    X = df['text']
    y = df['label']
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)
    model.fit(X_vec, y)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter the news article content below:")

user_input = st.text_area("News Content", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 'FAKE':
            st.error("🚨 The news is predicted to be FAKE.")
        else:
            st.success("✅ The news is predicted to be REAL.")
