import streamlit as st
import joblib
import os
import pdfplumber

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

skills_list = [
    "python", "machine learning", "data science", "sql",
    "tensorflow", "java", "spring", "excel",
    "communication", "leadership", "analysis"
]

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found

st.set_page_config(page_title="AI Resume Screener", page_icon="🤖")

st.title("🤖 AI Resume Screener (Pro Version)")

st.markdown("---")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

job = st.text_area("💼 Paste Job Description")

if st.button("🔍 Analyze Resume"):

    if uploaded_file is None or job.strip() == "":
        st.warning("⚠️ Upload resume and enter job description")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)

        combined = resume_text + " " + job.lower()

        vector = vectorizer.transform([combined])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1] * 100

        st.subheader("📊 Match Result")

        if prediction == 1:
            st.success(f"✅ Good Match ({probability:.2f}%)")
        else:
            st.error(f"❌ Not a Good Match ({probability:.2f}%)")

        st.progress(int(probability))
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job.lower())

        matched = list(set(resume_skills) & set(job_skills))
        missing = list(set(job_skills) - set(resume_skills))

        st.markdown("### 🧠 Skill Analysis")

        st.write("✅ Matched Skills:", matched if matched else "None")
        st.write("❌ Missing Skills:", missing if missing else "None")

        st.markdown("### 📌 Insight")
        if probability > 75:
            st.write("Strong candidate match.")
        elif probability > 50:
            st.write("Moderate match. Improve missing skills.")
        else:
            st.write("Low match. Candidate not suitable.")
