import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Remove missing values and duplicates
df = df.dropna()
df = df.drop_duplicates()

# Convert labels into numbers
df['label'] = df['label'].replace({'ham': 0, 'spam': 1}).astype(int)

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)

# Convert text into numbers using TF-IDF
X = df['text']
y = df['label'].astype(int)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# Step 13: Save model and vectorizer
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/vectorizer.pkl")

print("\nModel saved successfully!")
print("Saved file: model/spam_model.pkl")
print("Saved file: model/vectorizer.pkl")