from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]

        cleaned = clean_input(news_text)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        # Correct label mapping
        proba = model.predict_proba(vector)[0]
        confidence = round(max(proba) * 100, 2)
        prediction = "FAKE NEWS 🔴" if result == 0 else "REAL NEWS 🟢"

    return render_template(
    "index.html",
    prediction=prediction,
    news_text=news_text,
    confidence=confidence
)


if __name__ == "__main__":
    app.run(debug=True)
