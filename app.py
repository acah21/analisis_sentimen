# ===============================
# Dashboard Analisis Sentimen YouTube
# TF-IDF + XGBoost
# ===============================

import streamlit as st
import pandas as pd
import re, emoji, joblib, base64
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
# REMOVE STREAMLIT PADDING
# ===============================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        padding: 0;
    }
    [data-testid="stHeader"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOAD BACKGROUND IMAGE
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
# SIDEBAR BUTTONS
# ===============================
st.sidebar.title("📌 Menu")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"

if st.sidebar.button("🎥 Analisis YouTube"):
    st.session_state.page = "youtube"

if st.sidebar.button("📝 Analisis Kalimat"):
    st.session_state.page = "kalimat"

# ===============================
# HOME PAGE
# ===============================
if st.session_state.page == "home":

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
        .box {{
            background: rgba(0,0,0,0.55);
            padding: 80px;
            border-radius: 20px;
            color: white;
            text-align: center;
        }}
        </style>

        <div class="hero">
            <div class="box">
                <h1>Dashboard Analisis Sentimen YouTube</h1>
                <p>
                    Analisis komentar YouTube & kalimat teks<br>
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
elif st.session_state.page == "youtube":

    st.title("🎥 Analisis Sentimen Komentar YouTube")
    link = st.text_input("Masukkan link YouTube")

    if st.button("📊 Analisis"):
        video_id = extract_video_id(link)

        if video_id is None:
            st.error("Link YouTube tidak valid")
        else:
            with st.spinner("Memproses komentar..."):
                comments = get_comments(video_id)
                df = pd.DataFrame(comments, columns=["comment"])
                df["clean"] = df["comment"].apply(preprocess_text)

                X = tfidf.transform(df["clean"])
                df["sentiment"] = model.predict(X)

            st.success("Analisis selesai")

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"

# ===============================
# ANALISIS KALIMAT
# ===============================
elif st.session_state.page == "kalimat":

    st.title("📝 Analisis Sentimen Kalimat")
    kalimat = st.text_area("Masukkan kalimat")

    if st.button("🔍 Analisis Kalimat"):
        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        st.info(f"Kalimat: {kalimat}")
        st.success("Sentimen POSITIF" if pred == 1 else "Sentimen NEGATIF")

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"
