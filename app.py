# ===============================
# Dashboard Analisis Sentimen YouTube
# TF-IDF + XGBoost
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re, emoji, joblib
import matplotlib.pyplot as plt

from googleapiclient.discovery import build
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    layout="wide"
)

# ===============================
# DOWNLOAD NLTK (AMAN DI STREAMLIT)
# ===============================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model_xgboost_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# PREPROCESSING
# ===============================
stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

normalisasi_dict = {
    "gk": "tidak", "ga": "tidak", "ngga": "tidak",
    "yg": "yang", "d": "di", "klo": "kalau",
    "gw": "saya", "gue": "saya", "km": "kamu", "tp": "tapi"
}

def preprocess_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [normalisasi_dict.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

# ===============================
# YOUTUBE FUNCTIONS
# ===============================
API_KEY = "AIzaSyCz4yYoK_w-3IYkbeGCHtL4CoATBf6VN1I"

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        return None

def get_comments(video_id, max_results=500):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===============================
# HOME PAGE
# ===============================
if st.session_state.page == "home":

    st.markdown(
        """
        <style>
        .hero {
            background-image: url("bg.jpeg");
            background-size: cover;
            background-position: center;
            padding: 160px 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Dashboard Analisis Sentimen YouTube</h1>
            <h3>TF-IDF + XGBoost</h3>
            <p>Menganalisis komentar video YouTube secara otomatis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    if st.button("🔍 Mulai Analisis Sentimen"):
        st.session_state.page = "input"

# ===============================
# INPUT PAGE
# ===============================
elif st.session_state.page == "input":

    st.title("📥 Input Data Analisis")

    yt_link = st.text_input("Masukkan Link YouTube")
    kalimat = st.text_area("Masukkan Kalimat (opsional)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Analisis YouTube"):
            st.session_state.yt_link = yt_link
            st.session_state.page = "hasil"

    with col2:
        if st.button("📝 Analisis Kalimat"):
            clean = preprocess_text(kalimat)
            X = tfidf.transform([clean])
            pred = model.predict(X)[0]
            st.session_state.single_result = "Positif" if pred == 1 else "Negatif"
            st.session_state.page = "hasil"

    st.write("")
    if st.button("⬅️ Kembali"):
        st.session_state.page = "home"

# ===============================
# HASIL PAGE
# ===============================
elif st.session_state.page == "hasil":

    st.title("📈 Hasil Analisis Sentimen")

    # ----- HASIL KALIMAT -----
    if "single_result" in st.session_state:
        st.subheader("Hasil Analisis Kalimat")
        st.success(f"Sentimen: **{st.session_state.single_result}**")

    # ----- HASIL YOUTUBE -----
    if "yt_link" in st.session_state:
        video_id = extract_video_id(st.session_state.yt_link)

        comments = get_comments(video_id)
        df = pd.DataFrame(comments, columns=["comment"])
        df["clean"] = df["comment"].apply(preprocess_text)

        X = tfidf.transform(df["clean"])
        df["label"] = model.predict(X)
        df["sentiment"] = df["label"].map({1: "Positif", 0: "Negatif"})

        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        st.subheader("Top 5 Komentar Positif")
        st.write(df[df["sentiment"] == "Positif"]["comment"].head(5))

        st.subheader("Top 5 Komentar Negatif")
        st.write(df[df["sentiment"] == "Negatif"]["comment"].head(5))

    st.write("")
    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"
