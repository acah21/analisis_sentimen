# =========================================
# DASHBOARD ANALISIS SENTIMEN KOMENTAR YOUTUBE
# Model: TF-IDF + XGBoost
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import re, emoji, joblib
import matplotlib.pyplot as plt

from googleapiclient.discovery import build
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# =========================================
# KONFIGURASI STREAMLIT
# =========================================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Dashboard Analisis Sentimen Komentar YouTube")
st.write(
    "Aplikasi ini menganalisis sentimen komentar YouTube "
    "menggunakan **TF-IDF + XGBoost** (Positif & Negatif)."
)

# =========================================
# DOWNLOAD RESOURCE NLTK (AMAN UNTUK DEPLOY)
# =========================================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =========================================
# LOAD MODEL & TF-IDF
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("model_xgboost_sentiment.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# =========================================
# PREPROCESSING
# =========================================
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

# =========================================
# FUNGSI EKSTRAK VIDEO ID
# =========================================
def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        return None

# =========================================
# FUNGSI AMBIL KOMENTAR YOUTUBE
# =========================================
API_KEY = "AIzaSyCz4yYoK_w-3IYkbeGCHtL4CoATBf6VN1I"

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

# =========================================
# INPUT USER
# =========================================
st.subheader("🔗 Input Link YouTube")
video_link = st.text_input("Masukkan link video YouTube")

# =========================================
# PROSES ANALISIS
# =========================================
if st.button("🚀 Analisis Sentimen"):
    if not video_link:
        st.warning("Masukkan link YouTube terlebih dahulu.")
        st.stop()

    video_id = extract_video_id(video_link)
    if video_id is None:
        st.error("Link YouTube tidak valid.")
        st.stop()

    with st.spinner("Mengambil dan menganalisis komentar..."):
        comments = get_comments(video_id, max_results=500)

        df = pd.DataFrame(comments, columns=["comment"])
        df["clean_comment"] = df["comment"].apply(preprocess_text)

        X = tfidf.transform(df["clean_comment"])
        df["label"] = model.predict(X)
        df["sentiment"] = df["label"].map({1: "Positif", 0: "Negatif"})

    # =========================================
    # OUTPUT
    # =========================================
    st.success(f"Berhasil menganalisis {len(df)} komentar")

    st.subheader("📄 Contoh Hasil Analisis")
    st.dataframe(df.head())

    # Pie Chart
    st.subheader("📊 Distribusi Sentimen")
    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot.pie(
        autopct="%1.1f%%",
        figsize=(5,5),
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    # Top komentar
    st.subheader("👍 Top 5 Komentar Positif")
    st.write(df[df["sentiment"]=="Positif"]["comment"].head(5))

    st.subheader("👎 Top 5 Komentar Negatif")
    st.write(df[df["sentiment"]=="Negatif"]["comment"].head(5))
