import streamlit as st
st.write("✅ Streamlit is rendering")  # ← ADD THIS LINE
import pandas as pd
import re

st.title("AI Resume Matcher – Text Preprocessing")

st.write("Paste your resume and the job description below:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_text = st.text_area("Resume Text", height=300)

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Job Description Text", height=300)


def clean_text(text: str) -> pd.DataFrame:
    # lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # tokenize
    tokens = text.split()

    # remove short tokens
    tokens = [t for t in tokens if len(t) > 2]

    df = pd.DataFrame(tokens, columns=["token"])
    return df

if st.button("Clean Text"):
    if resume_text.strip() == "" or jd_text.strip() == "":
        st.warning("Please fill in both the resume and job description.")
    else:
        resume_df = clean_text(resume_text)
        jd_df = clean_text(jd_text)

        st.subheader("Cleaned Resume Tokens")
        st.dataframe(resume_df)

        st.subheader("Cleaned Job Description Tokens")
        st.dataframe(jd_df)
