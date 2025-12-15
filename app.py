# ===============================
# Dashboard Analisis Sentimen YouTube
# TF-IDF + XGBoost
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re, emoji, joblib
import matplotlib.pyplot as plt
import base64

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
# LOAD IMAGE BASE64
# ===============================
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64_image("bg.jpeg")

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
# YOUTUBE FUNCTION
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
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("📌 Menu")

menu = st.sidebar.radio(
    "Navigasi",
    ["Home", "Analisis YouTube", "Analisis Kalimat"]
)

# ===============================
# HOME PAGE (FULL SCREEN)
# ===============================
if menu == "Home":

    st.markdown(
        f"""
        <style>
        .hero {{
            height: 100vh;
            background-image: url("data:image/jpeg;base64,{bg}");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .content {{
            background: rgba(0,0,0,0.55);
            padding: 70px;
            border-radius: 20px;
            color: white;
            text-align: center;
            width: 85%;
        }}
        .content h1 {{
            font-size: 3.2rem;
        }}
        </style>

        <div class="hero">
            <div class="content">
                <h1>Dashboard Analisis Sentimen YouTube</h1>
                <p>
                    Analisis komentar YouTube dan kalimat teks<br>
                    menggunakan TF-IDF dan XGBoost
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# ANALISIS YOUTUBE
# ===============================
elif menu == "Analisis YouTube":

    st.title("🎥 Analisis Sentimen Komentar YouTube")

    link = st.text_input("Masukkan link YouTube")

    if st.button("📊 Analisis Komentar"):
        video_id = extract_video_id(link)

        if video_id is None:
            st.error("Link YouTube tidak valid")
        else:
            with st.spinner("Mengambil dan menganalisis komentar..."):
                comments = get_comments(video_id)
                df = pd.DataFrame(comments, columns=["comment"])
                df["clean"] = df["comment"].apply(preprocess_text)

                X = tfidf.transform(df["clean"])
                df["label"] = model.predict(X)
                df["sentiment"] = df["label"].map({1: "Positif", 0: "Negatif"})

            st.success("Analisis selesai")

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

# ===============================
# ANALISIS KALIMAT
# ===============================
elif menu == "Analisis Kalimat":

    st.title("📝 Analisis Sentimen Kalimat")

    kalimat = st.text_area("Masukkan kalimat yang ingin dianalisis")

    if st.button("🔍 Analisis Kalimat"):
        if kalimat.strip() == "":
            st.warning("Kalimat tidak boleh kosong")
        else:
            clean = preprocess_text(kalimat)
            X = tfidf.transform([clean])
            pred = model.predict(X)[0]

            st.info(f"Kalimat: {kalimat}")
            st.success("Sentimen POSITIF" if pred == 1 else "Sentimen NEGATIF")
