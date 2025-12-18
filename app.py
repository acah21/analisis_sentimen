# =====================================
# Dashboard Analisis Sentimen YouTube
# TF-IDF + XGBoost
# =====================================

import streamlit as st
import pandas as pd
import re, emoji, joblib, base64
import matplotlib.pyplot as plt

from googleapiclient.discovery import build
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    layout="wide"
)

# =====================================
# LOAD BACKGROUND IMAGE
# =====================================
def load_bg_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = load_bg_image("bg.jpeg")

# =====================================
# NLTK SAFE LOAD
# =====================================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("model_xgboost_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# =====================================
# PREPROCESSING
# =====================================
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

# =====================================
# YOUTUBE FUNCTIONS
# =====================================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

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

# =====================================
# HERO SECTION (HOMEPAGE)
# =====================================
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
    }}
    .hero {{
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hero-box {{
        background: rgba(0,0,0,0.6);
        padding: 60px;
        border-radius: 20px;
        color: white;
        text-align: center;
        width: 70%;
    }}
    </style>

    <div class="hero">
        <div class="hero-box">
            <h1>Analisis Sentimen Komentar YouTube</h1>
            <p style="font-size:18px;">
                Sistem ini menganalisis sentimen komentar YouTube menggunakan
                metode <b>TF-IDF</b> dan algoritma <b>XGBoost</b>.
            </p>
            <p>Masukkan link video YouTube untuk mengetahui kecenderungan sentimen publik.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================
# INPUT LINK (CENTER)
# =====================================
st.markdown("## 🔗 Masukkan Link Video YouTube")
link = st.text_input("", placeholder="https://www.youtube.com/watch?v=...")

analyze = st.button("📊 Analisis Komentar")

# =====================================
# ANALISIS & OUTPUT
# =====================================
if analyze:
    video_id = extract_video_id(link)

    if not video_id:
        st.error("Link YouTube tidak valid")
    else:
        with st.spinner("Mengambil dan menganalisis komentar..."):
            comments = get_comments(video_id)
            df = pd.DataFrame(comments, columns=["comment"])
            df["clean"] = df["comment"].apply(preprocess_text)

            X = tfidf.transform(df["clean"])
            df["sentiment"] = model.predict(X)
            df["sentiment"] = df["sentiment"].map({1: "Positif", 0: "Negatif"})

        st.success("Analisis selesai")

        # ===============================
        # HASIL ANALISIS
        # ===============================
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Sentimen")

        sentiment_count = df["sentiment"].value_counts()
        total = sentiment_count.sum()

        positif_pct = (sentiment_count.get("Positif", 0) / total) * 100
        negatif_pct = (sentiment_count.get("Negatif", 0) / total) * 100

        col1, col2 = st.columns([1, 2])

        with col1:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            sentiment_count.plot.pie(
                autopct="%1.1f%%",
                startangle=90,
                ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

        with col2:
            st.markdown("### 🧾 Interpretasi Hasil")
            st.write(
                f"""
                Dari **{total} komentar** yang dianalisis:
                - **{positif_pct:.2f}%** bersentimen **positif**
                - **{negatif_pct:.2f}%** bersentimen **negatif**

                Hal ini menunjukkan bahwa sentimen publik terhadap video
                cenderung **{"positif" if positif_pct > negatif_pct else "negatif"}**.
                """
            )

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Top 5 Komentar Positif**")
            st.write(df[df["sentiment"] == "Positif"]["comment"].head(5))

        with col4:
            st.markdown("**Top 5 Komentar Negatif**")
            st.write(df[df["sentiment"] == "Negatif"]["comment"].head(5))
