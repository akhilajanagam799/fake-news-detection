# 📰 Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-orange?logo=streamlit)

---

## 📌 Project Overview

The **Fake News Detection System** is an intelligent application built using **Machine Learning** and **Natural Language Processing (NLP)** to automatically classify news articles as **Real** or **Fake**. The aim is to combat misinformation by providing a quick way to verify the authenticity of a news article.

This project includes:
- Clean model training using TF-IDF & Logistic Regression
- A lightweight, interactive web application using **Streamlit**
- Optimized architecture with a modular and scalable structure

---

## 🎯 Key Features

- 🔍 Real-time classification of custom news text
- 🧠 Trained ML model using **TF-IDF** and **Logistic Regression**
- 🧾 Organized, modular codebase and directory structure
- 🚀 Simple, elegant **Streamlit** UI for interactive testing
- 🧪 Tested on a real-world dataset with high accuracy

---

## 🛠️ Tech Stack

| Category        | Technologies Used                        |
|-----------------|------------------------------------------|
| **Programming** | Python 3.10                              |
| **Libraries**   | pandas, numpy, scikit-learn, streamlit   |
| **ML Model**    | TF-IDF Vectorization + Logistic Regression |
| **Deployment**  | Streamlit Cloud                          |
| **Version Control** | Git & GitHub                         |

---

## 🧾 Project Structure

📦 fake-news-detection
├── app/


│ └── fake_news_app.py # Streamlit frontend code

├── data/

│ └── [Large dataset files ignored in .gitignore]

├── models/

│ ├── fake_news_model.pkl # Trained ML model

│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer

├── main.py # Script for training and model building

├── requirements.txt # Python dependencies

└── README.md # Project documentation


---

## 🧠 How It Works

1. **Preprocessing**: News text is cleaned and vectorized using TF-IDF.
2. **Modeling**: A **Logistic Regression** classifier is trained on labeled real/fake news data.
3. **Prediction**: The model predicts the probability of the news being real or fake.
4. **Frontend**: Users can input any news content in the Streamlit app and instantly get results.

---

## 📊 Dataset

> Source: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- `Fake.csv`: Fake news articles
- `True.csv`: Real news articles  
✅ *(Both files are large, and excluded from GitHub using `.gitignore`)*

---

## 💻 Run Locally


Step 1: Clone the repository
```bash
git clone https://github.com/akhilajanagam799/fake-news-detection.git
cd fake-news-detection
```

Step 2: (Optional) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

#Step 3: Install required packages
```bash
pip install -r requirements.txt
```

# Step 4: Launch the Streamlit App
streamlit run app/fake_news_app.py




