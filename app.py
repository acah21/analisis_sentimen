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
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    layout="wide"
)

# ===============================
# NLTK SAFE DOWNLOAD
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
API_KEY = "ISI_API_KEY_KAMU"

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def get_comments(video_id, max_results=300):
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
if "menu" not in st.session_state:
    st.session_state.menu = "home"

# ===============================
# HOME PAGE (MOUNT JAWA STYLE)
# ===============================
if st.session_state.menu == "home":

    st.markdown(
        """
        <style>
        .hero {
            position: relative;
            height: 85vh;
            background-image: url("bg.jpeg");
            background-size: cover;
            background-position: center;
            border-radius: 20px;
        }
        .overlay {
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.5);
            border-radius: 20px;
        }
        .hero-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: white;
        }
        .hero h1 {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <div class="overlay"></div>
            <div class="hero-content">
                <h1>Dashboard Analisis Sentimen YouTube</h1>
                <p>TF-IDF + XGBoost untuk Analisis Komentar</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎥 Analisis YouTube"):
            st.session_state.menu = "youtube"
            st.rerun()

    with col2:
        if st.button("📝 Analisis Kalimat"):
            st.session_state.menu = "kalimat"
            st.rerun()

# ===============================
# ANALISIS YOUTUBE
# ===============================
elif st.session_state.menu == "youtube":

    st.title("🎥 Analisis Sentimen YouTube")

    link = st.text_input("Masukkan Link YouTube")

    if st.button("📊 Analisis"):
        video_id = extract_video_id(link)

        if video_id is None:
            st.error("Link YouTube tidak valid")
        else:
            with st.spinner("Mengambil & menganalisis komentar..."):
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
            st.write(df[df["sentiment"]=="Positif"]["comment"].head(5))

            st.subheader("Top 5 Komentar Negatif")
            st.write(df[df["sentiment"]=="Negatif"]["comment"].head(5))

    if st.button("⬅️ Kembali"):
        st.session_state.menu = "home"
        st.rerun()

# ===============================
# ANALISIS KALIMAT
# ===============================
elif st.session_state.menu == "kalimat":

    st.title("📝 Analisis Sentimen Kalimat")

    kalimat = st.text_area("Masukkan kalimat yang ingin dianalisis")

    if st.button("🔍 Analisis Kalimat"):
        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        st.subheader("Hasil Analisis")
        st.info(f"Kalimat: {kalimat}")
        st.success("Sentimen: POSITIF" if pred == 1 else "Sentimen: NEGATIF")

    if st.button("⬅️ Kembali"):
        st.session_state.menu = "home"
        st.rerun()
