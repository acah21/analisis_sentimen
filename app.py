# ===============================
# Dashboard Analisis Sentimen YouTube
# TF-IDF + XGBoost
# ===============================

import streamlit as st
import pandas as pd
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
# DOWNLOAD NLTK
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
    "yg": "yang", "klo": "kalau", "gw": "saya", "gue": "saya"
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
API_KEY = st.secrets["API_KEY"]

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
# NAVIGASI
# ===============================
menu = st.session_state.get("menu", "home")

# ===============================
# HOME
# ===============================
if menu == "home":

    st.image("bg.jpeg", use_container_width=True)

    st.markdown(
        "<h1 style='text-align:center;'>Dashboard Analisis Sentimen YouTube</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;'>TF-IDF + XGBoost</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎥 Analisis YouTube", use_container_width=True):
            st.session_state.menu = "youtube"
            st.rerun()

    with col2:
        if st.button("📝 Analisis Kalimat", use_container_width=True):
            st.session_state.menu = "kalimat"
            st.rerun()

# ===============================
# ANALISIS YOUTUBE
# ===============================
elif menu == "youtube":

    st.title("🎥 Analisis Sentimen Komentar YouTube")

    yt_link = st.text_input("Masukkan Link YouTube")

    if st.button("📊 Analisis YouTube"):
        video_id = extract_video_id(yt_link)

        if not video_id:
            st.error("Link YouTube tidak valid.")
        else:
            with st.spinner("Mengambil dan menganalisis komentar..."):
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

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.menu = "home"
        st.rerun()

# ===============================
# ANALISIS KALIMAT
# ===============================
elif menu == "kalimat":

    st.title("📝 Analisis Sentimen Kalimat")

    kalimat = st.text_area("Masukkan Kalimat")

    if st.button("🔍 Analisis Kalimat"):
        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]
        hasil = "Positif" if pred == 1 else "Negatif"

        st.info(f"Kalimat yang dianalisis:\n\n{kalimat}")
        st.success(f"Hasil Sentimen: **{hasil}**")

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.menu = "home"
        st.rerun()
