# 📧 SMS & Email Spam Classifier (NLP & ML System)

An end-to-end Natural Language Processing (NLP) and Machine Learning application designed to classify emails and SMS messages as **Spam** or **Ham (Legitimate)** in real time.

---

## 📌 Project Overview
Spam emails pose security threats and clutter user communication. This project processes raw text messages using text preprocessing and TF-IDF vectorization, trains high-precision classification models, and provides an interactive web interface using **Streamlit**.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Libraries:** Scikit-Learn, NLTK, Pandas, NumPy, Joblib, Streamlit
- **Algorithms:** Multinomial Naive Bayes, Logistic Regression, Random Forest

---

## 📂 Project Structure
```text
Spam-Email-Classifier/
│
├── dataset/
│   └── spam.csv
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── Screenshot/
│   ├── home.png
│   └── Half code with side by run .png
├── app.py
├── train.py
├── requirements.txt
└── README.md
