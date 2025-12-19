import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# first time only
# nltk.download('stopwords')

# -----------------------------
# STEP 7: Load CLEAN dataset
# -----------------------------
data = pd.read_csv("dataset/fake_news_clean.csv")

# -----------------------------
# Text preprocessing
# -----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(preprocess_text)

print("✅ Step 7 done: Text preprocessing complete")

# -----------------------------
# STEP 8: TF-IDF Vectorization
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

print("✅ Step 8 done: TF-IDF feature extraction complete")
print("Feature matrix shape:", X.shape)

joblib.dump(X, "dataset/tfidf_features.pkl")
joblib.dump(y, "dataset/labels.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("💾 All outputs saved successfully")