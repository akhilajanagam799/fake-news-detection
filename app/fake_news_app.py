import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("📰 Fake News Detection App")
st.write("Enter news content below to check if it's **Real** or **Fake**.")

input_text = st.text_area("Paste your news article here:")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    confidence = round(max(proba) * 100, 2)

    if prediction.lower() == "fake":
        st.success(f"✅ This news is **FAKE**.")
    else:
        st.success(f"✅ This news is **REAL**.")
    
    st.info(f"🔍 Confidence: {confidence}%")
