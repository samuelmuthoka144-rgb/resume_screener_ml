import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    df = df.dropna()

    df['resume_text'] = df['resume_text'].apply(clean_text)
    df['job_description'] = df['job_description'].apply(clean_text)

    return df

if __name__ == "__main__":
    path = "../data/processed/resume_dataset.csv"

    df = pd.read_csv(path)

    df = preprocess_data(df)

    print(df.head())