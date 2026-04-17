import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_engineering import load_data, combine_text, extract_features


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "processed", "resume_dataset.csv")

    df = load_data(path)
    df = combine_text(df)

    X, y, vectorizer = extract_features(df)

    if len(set(y)) < 2:
        raise ValueError("Dataset must contain BOTH classes (0 and 1). Fix your CSV.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n===== FINAL RESULTS =====")
    print("Accuracy:", round(accuracy, 3))

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

    print("Model saved successfully 🚀")