import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load dataset
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine and shuffle
df = pd.concat([fake, real])
df = df[['text', 'label']].sample(frac=1, random_state=42).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF vectorization
X = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("models/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
