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
# NLTK SAFE
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
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===============================
# SIDEBAR (NAVIGASI SAJA)
# ===============================
st.sidebar.title("🔎 Navigasi")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
    st.rerun()

if st.sidebar.button("🎥 Analisis YouTube"):
    st.session_state.page = "youtube"
    st.rerun()

if st.sidebar.button("📝 Analisis Kalimat"):
    st.session_state.page = "kalimat"
    st.rerun()

# ===============================
# HOMEPAGE
# ===============================
if st.session_state.page == "home":

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("bg.jpeg");
            background-size: cover;
            background-position: center;
        }
        .hero-box {
            background: rgba(255,255,255,0.85);
            padding: 50px;
            border-radius: 20px;
            margin-top: 120px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-box">
            <h1>Dashboard Analisis Sentimen YouTube</h1>
            <p>
            Aplikasi untuk menganalisis sentimen komentar YouTube menggunakan
            <b>TF-IDF</b> dan <b>XGBoost</b>.
            </p>
            <p>Gunakan menu di sidebar untuk memulai analisis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# ANALISIS YOUTUBE
# ===============================
elif st.session_state.page == "youtube":

    st.title("🎥 Analisis Sentimen Komentar YouTube")

    link = st.text_input("Masukkan Link Video YouTube")

    if st.button("📊 Analisis Komentar"):
        video_id = extract_video_id(link)

        if not video_id:
            st.error("Link YouTube tidak valid")
        else:
            with st.spinner("Mengambil dan menganalisis komentar..."):
                comments = get_comments(video_id)
                df = pd.DataFrame(comments, columns=["comment"])
                df["clean"] = df["comment"].apply(preprocess_text)

                X = tfidf.transform(df["clean"])
                df["label"] = model.predict(X)
                df["sentiment"] = df["label"].map({1: "Positif", 0: "Negatif"})

            # ===============================
            # HASIL
            # ===============================
            st.success("Analisis selesai")

            sentimen_count = df["sentiment"].value_counts(normalize=True) * 100
            positif = sentimen_count.get("Positif", 0)
            negatif = sentimen_count.get("Negatif", 0)

            st.subheader("📌 Interpretasi Hasil")
            st.write(
                f"""
                Dari seluruh komentar yang dianalisis:
                - **{positif:.2f}%** bersentimen **Positif**
                - **{negatif:.2f}%** bersentimen **Negatif**

                Hal ini menunjukkan bahwa respon audiens terhadap video tersebut
                cenderung **{'positif' if positif > negatif else 'negatif'}**.
                """
            )

            # Diagram kecil
            st.subheader("📊 Distribusi Sentimen")
            fig, ax = plt.subplots(figsize=(4, 4))
            df["sentiment"].value_counts().plot.pie(
                autopct="%1.1f%%", ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

            st.subheader("💬 Contoh Komentar Positif")
            st.write(df[df["sentiment"] == "Positif"]["comment"].head(5))

            st.subheader("💬 Contoh Komentar Negatif")
            st.write(df[df["sentiment"] == "Negatif"]["comment"].head(5))

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"
        st.rerun()

# ===============================
# ANALISIS KALIMAT
# ===============================
elif st.session_state.page == "kalimat":

    st.title("📝 Analisis Sentimen Kalimat")

    kalimat = st.text_area("Masukkan kalimat yang ingin dianalisis")

    if st.button("🔍 Analisis Kalimat"):
        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        st.subheader("📌 Hasil Analisis")
        st.write(f"**Kalimat:** {kalimat}")

        if pred == 1:
            st.success("Sentimen: POSITIF")
        else:
            st.error("Sentimen: NEGATIF")

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"
        st.rerun()
