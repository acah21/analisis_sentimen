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

# ==================================
# CONFIG
# ==================================
st.set_page_config(page_title="Analisis Sentimen YouTube", layout="wide")

# ==================================
# CSS BACKGROUND (MENIRU MOUNT JAWA)
# ==================================
homepage_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('bg.jpeg');
    background-size: cover;
    background-position: center;
}
</style>
"""

result_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: #ffffff !important;
}
</style>
"""

# ==================================
# STATE
# ==================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==================================
# NLTK SAFE
# ==================================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ==================================
# LOAD MODEL
# ==================================
model = joblib.load("model_xgboost_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ==================================
# PREPROCESSING
# ==================================
stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ==================================
# YOUTUBE
# ==================================
API_KEY = "ISI_API_KEY_KAMU"

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def get_comments(video_id, max_results=300):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments, next_page_token = [], None

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

# ==================================
# SIDEBAR (MENIRU MOUNT JAWA)
# ==================================
st.sidebar.header("Pilih Analisis")

yt_link = st.sidebar.text_input("Link YouTube")
kalimat = st.sidebar.text_area("Kalimat")

if st.sidebar.button("🎥 Analisis YouTube"):
    st.session_state.page = "yt"

if st.sidebar.button("📝 Analisis Kalimat"):
    st.session_state.page = "kalimat"

# ==================================
# PAGE 1: HOME
# ==================================
if st.session_state.page == "home":
    st.markdown(homepage_bg, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 40px;
        border-radius: 20px;
        margin-top: 120px;
        text-align: center;">
        <h1>Dashboard Analisis Sentimen YouTube</h1>
        <p>Menganalisis sentimen komentar YouTube & kalimat teks</p>
        <p>Gunakan menu di sidebar untuk memulai analisis</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================
# PAGE 2: ANALISIS YOUTUBE
# ==================================
elif st.session_state.page == "yt":
    st.markdown(result_bg, unsafe_allow_html=True)
    st.header("📊 Hasil Analisis Sentimen YouTube")

    video_id = extract_video_id(yt_link)

    if video_id:
        with st.spinner("Mengambil komentar..."):
            comments = get_comments(video_id)
            df = pd.DataFrame(comments, columns=["comment"])
            df["clean"] = df["comment"].apply(preprocess_text)
            df["label"] = model.predict(tfidf.transform(df["clean"]))
            df["sentiment"] = df["label"].map({1:"Positif",0:"Negatif"})

        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        st.subheader("Top 5 Komentar Positif")
        st.write(df[df["sentiment"]=="Positif"]["comment"].head())

        st.subheader("Top 5 Komentar Negatif")
        st.write(df[df["sentiment"]=="Negatif"]["comment"].head())

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"

# ==================================
# PAGE 3: ANALISIS KALIMAT
# ==================================
elif st.session_state.page == "kalimat":
    st.markdown(result_bg, unsafe_allow_html=True)
    st.header("📝 Hasil Analisis Kalimat")

    clean = preprocess_text(kalimat)
    pred = model.predict(tfidf.transform([clean]))[0]

    st.info(f"Kalimat yang dianalisis:\n\n{kalimat}")
    st.success("Sentimen POSITIF" if pred == 1 else "Sentimen NEGATIF")

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"

