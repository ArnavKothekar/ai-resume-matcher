import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# App sanity check
# -------------------------------
st.write("✅ Streamlit is rendering")

st.title("AI Resume Matcher – Text Preprocessing & Similarity")
st.write("Paste your resume and the job description below:")

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_text = st.text_area("Resume Text", height=300)

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Job Description Text", height=300)

# -------------------------------
# Text cleaning (for inspection)
# -------------------------------
def clean_text(text: str) -> pd.DataFrame:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return pd.DataFrame(tokens, columns=["token"])

# -------------------------------
# Token set for coverage
# -------------------------------
def important_token_set(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    tokens = [
        t for t in tokens
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]

    return set(tokens)

# -------------------------------
# Similarity computation
# -------------------------------
def compute_similarity(resume_text: str, jd_text: str):
    # --- TF-IDF cosine similarity ---
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        [resume_text, jd_text]
    )

    cosine_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    # --- Unique keyword coverage (JD → Resume) ---
    resume_tokens = important_token_set(resume_text)
    jd_tokens = important_token_set(jd_text)

    if jd_tokens:
        coverage_score = len(resume_tokens & jd_tokens) / len(jd_tokens)
    else:
        coverage_score = 0.0

    # --- Weighted aggregation ---
    FINAL_SCORE = (
        0.7 * cosine_score +
        0.8 * coverage_score
    )

    return cosine_score, coverage_score, FINAL_SCORE

# -------------------------------
# Button action
# -------------------------------
if st.button("Clean & Match"):
    if resume_text.strip() == "" or jd_text.strip() == "":
        st.warning("Please fill in both the resume and job description.")
    else:
        # Cleaned token views
        resume_df = clean_text(resume_text)
        jd_df = clean_text(jd_text)

        cosine_score, coverage_score, final_score = compute_similarity(
            resume_text, jd_text
        )

        # Display results
        st.subheader("Match Score")
        st.metric(
            label="Overall Resume ↔ Job Match",
            value=f"{final_score * 100:.2f}%"
        )

        with st.expander("Score Breakdown"):
            st.write(f"TF-IDF Cosine Similarity: {cosine_score * 100:.2f}%")
            st.write(f"Unique Keyword Coverage: {coverage_score * 100:.2f}%")

        st.subheader("Cleaned Resume Tokens")
        st.dataframe(resume_df)

        st.subheader("Cleaned Job Description Tokens")
        st.dataframe(jd_df)
