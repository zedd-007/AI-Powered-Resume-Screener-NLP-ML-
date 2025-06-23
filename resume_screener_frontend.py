# resume_screener_frontend.py

import streamlit as st
import os
import re
import nltk
import torch
import zipfile
import pickle
import tempfile
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from docx import Document
from PyPDF2 import PdfReader

# Load models
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_classifier.pkl", 'rb') as f:
    classifier = pickle.load(f)
with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_model.pkl", 'rb') as f:
    bert_model = pickle.load(f)

# Helpers
def extract_text(file, ext):
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "pdf":
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == "docx":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def predict_resume_category(text):
    clean = preprocess_text(text)
    embedding = get_embedding(clean).reshape(1, -1)
    return classifier.predict(embedding)[0], embedding

def get_similarity_score(embedding1, embedding2):
    return float(cosine_similarity([embedding1], [embedding2])[0][0])

# === Streamlit UI ===
st.title("🤖 AI-Powered Resume Screener + Match Scoring")

st.markdown("Upload a resume file and compare it to a Job Description (JD) for a match score.")

# --- Job Description ---
st.subheader("🧾 Paste Job Description")
job_description = st.text_area("Enter the job description here...", height=150)
job_embedding = None

if job_description:
    job_cleaned = preprocess_text(job_description)
    job_embedding = get_embedding(job_cleaned)

# === Single Resume ===
st.header("📄 Single Resume Screening")
single_file = st.file_uploader("Upload Resume File (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if single_file:
    ext = single_file.name.split('.')[-1].lower()
    resume_text = extract_text(single_file, ext)
    st.text_area("📄 Resume Preview", resume_text[:800] + '...', height=200)

    if st.button("🔍 Predict & Score"):
        label, emb = predict_resume_category(resume_text)
        st.success(f"🧠 Predicted Job Domain: **{label}**")

        if job_embedding is not None:
            score = get_similarity_score(emb, job_embedding)
            st.info(f"🔗 Match Score with Job Description: **{score:.2f}**")
        else:
            st.warning("No job description provided. Match score skipped.")

# === Bulk Resume Screening ===
st.header("📁 Bulk Resume Screening (ZIP)")

bulk_zip = st.file_uploader("Upload ZIP of resumes", type=["zip"])

if bulk_zip:
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(bulk_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        results = []

        for file_name in os.listdir(temp_dir):
            ext = file_name.split('.')[-1].lower()
            if ext not in ['txt', 'pdf', 'docx']:
                continue

            try:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, 'rb') as f:
                    resume_text = extract_text(f, ext)
                    label, emb = predict_resume_category(resume_text)
                    score = get_similarity_score(emb, job_embedding) if job_embedding is not None else "N/A"
                    results.append((file_name, label, score))
            except Exception as e:
                results.append((file_name, "Error", str(e)))

        st.subheader("📊 Bulk Screening Results")
        for file, label, score in results:
            score_display = f"{score:.2f}" if isinstance(score, float) else score
            st.write(f"📄 **{file}** ➤ **{label}** | Match Score: **{score_display}**")