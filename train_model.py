import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load saved TF-IDF features
# -----------------------------
X = joblib.load("dataset/tfidf_features.pkl")
y = joblib.load("dataset/labels.pkl")

print("✅ Features loaded successfully")
print("Feature shape:", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Train-test split done")

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Model training complete")

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, "model/model.pkl")

print("💾 Model saved as model/model.pkl")
