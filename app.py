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
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    layout="wide"
)

# ===============================
# BACKGROUND IMAGE (GITHUB RAW)
# GANTI DENGAN LINK BG.JPEG KAMU
# ===============================
HOMEPAGE_BG = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/USERNAME/REPO/main/bg.jpeg");
    background-size: cover;
    background-position: center;
}
</style>
"""

NORMAL_BG = """
<style>
[data-testid="stAppViewContainer"] {
    background: #ffffff;
}
</style>
"""

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
# SIDEBAR (NAVIGASI)
# ===============================
st.sidebar.title("Navigasi")

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
# HOME PAGE
# ===============================
if st.session_state.page == "home":

    st.markdown(HOMEPAGE_BG, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.88);
        padding: 50px;
        border-radius: 20px;
        margin-top: 120px;
        text-align: center;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        <h1>Dashboard Analisis Sentimen YouTube</h1>
        <p>
        Aplikasi untuk menganalisis sentimen komentar YouTube
        menggunakan <b>TF-IDF</b> dan <b>XGBoost</b>.
        </p>
        <p>Gunakan menu di sidebar untuk memulai analisis.</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# ANALISIS YOUTUBE
# ===============================
elif st.session_state.page == "youtube":

    st.markdown(NORMAL_BG, unsafe_allow_html=True)
    st.title("🎥 Analisis Sentimen Komentar YouTube")

    link = st.text_input("Masukkan link video YouTube")

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

            # SIMPAN HASIL
            st.session_state.yt_result = df

    # TAMPILKAN HASIL
    if "yt_result" in st.session_state:
        df = st.session_state.yt_result

        st.subheader("📊 Distribusi Sentimen")

        fig, ax = plt.subplots(figsize=(4, 4))
        df["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        positif = (df["sentiment"] == "Positif").mean() * 100
        negatif = (df["sentiment"] == "Negatif").mean() * 100

        st.info(
            f"""
            Dari komentar yang dianalisis:
            - **{positif:.2f}% Positif**
            - **{negatif:.2f}% Negatif**

            Mayoritas sentimen cenderung **{'positif' if positif > negatif else 'negatif'}**.
            """
        )

# ===============================
# ANALISIS KALIMAT
# ===============================
elif st.session_state.page == "kalimat":

    st.markdown(NORMAL_BG, unsafe_allow_html=True)
    st.title("📝 Analisis Sentimen Kalimat")

    kalimat = st.text_area("Masukkan kalimat")

    if st.button("🔍 Analisis Kalimat"):
        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        st.subheader("Hasil Analisis")
        st.write(f"**Kalimat:** {kalimat}")

        if pred == 1:
            st.success("Sentimen: POSITIF")
        else:
            st.error("Sentimen: NEGATIF")
