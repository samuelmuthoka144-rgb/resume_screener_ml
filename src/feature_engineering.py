import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    return df


def combine_text(df):
    required_cols = ['resume_text', 'job_description', 'label']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in dataset: {col}")

    df = df.dropna(subset=required_cols)
    df['combined_text'] = df['resume_text'].astype(str) + " " + df['job_description'].astype(str)

    return df


def extract_features(df):
    vectorizer = TfidfVectorizer(stop_words='english')

    X = vectorizer.fit_transform(df['combined_text'])
    y = df['label'].astype(int)

    return X, y, vectorizer


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "processed", "resume_dataset.csv")

    df = load_data(path)

    df = combine_text(df)

    X, y, vectorizer = extract_features(df)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Dataset loaded and processed successfully 🚀")