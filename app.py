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
# NLTK
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

# ===============================
# YOUTUBE
# ===============================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def get_comments(video_id, max_results=300):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page = None

    while len(comments) < max_results:
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page,
            textFormat="plainText"
        ).execute()

        for item in res["items"]:
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )

        next_page = res.get("nextPageToken")
        if not next_page:
            break

    return comments

# ===============================
# SESSION
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===============================
# HOME
# ===============================
if st.session_state.page == "home":

    st.markdown("""
    <style>
    .hero {
        background-image: url("bg.jpeg");
        background-size: cover;
        background-position: center;
        height: 80vh;
        border-radius: 24px;
        position: relative;
    }
    .overlay {
        background: rgba(0,0,0,0.55);
        height: 100%;
        border-radius: 24px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="overlay">
            <div>
                <h1>Analisis Sentimen YouTube</h1>
                <p>TF-IDF + XGBoost</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Mulai Analisis"):
        st.session_state.page = "input"

# ===============================
# INPUT
# ===============================
elif st.session_state.page == "input":

    st.title("📥 Input Data Analisis")

    yt_link = st.text_input("🔗 Masukkan Link YouTube")
    if st.button("📊 Analisis YouTube"):
        st.session_state.yt_link = yt_link
        st.session_state.page = "hasil"

    st.markdown("---")

    kalimat = st.text_area("📝 Masukkan Kalimat")
    if st.button("🧠 Analisis Kalimat"):
        st.session_state.input_kalimat = kalimat

        clean = preprocess_text(kalimat)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        st.session_state.single_result = "Positif" if pred == 1 else "Negatif"
        st.session_state.page = "hasil"

    if st.button("⬅️ Kembali ke Home"):
        st.session_state.page = "home"

# ===============================
# HASIL
# ===============================
elif st.session_state.page == "hasil":

    st.title("📈 Hasil Analisis Sentimen")

    # ===== HASIL KALIMAT =====
    if "single_result" in st.session_state:
        st.subheader("Hasil Analisis Kalimat")

        st.info(
            f"""
            **Kalimat yang dianalisis:**

            > {st.session_state.input_kalimat}
            """
        )

        st.success(
            f"""
            **Hasil Sentimen:**  
            👉 **{st.session_state.single_result}**

            Artinya, berdasarkan model yang digunakan, kalimat tersebut
            mengandung sentimen **{st.session_state.single_result.lower()}**.
            """
        )

    # ===== HASIL YOUTUBE =====
    if "yt_link" in st.session_state:
        vid = extract_video_id(st.session_state.yt_link)
        comments = get_comments(vid)

        df = pd.DataFrame(comments, columns=["comment"])
        df["clean"] = df["comment"].apply(preprocess_text)

        X = tfidf.transform(df["clean"])
        df["label"] = model.predict(X)
        df["sentiment"] = df["label"].map({1: "Positif", 0: "Negatif"})

        st.subheader("Distribusi Sentimen Komentar YouTube")
        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    if st.button("⬅️ Kembali ke Input"):
        st.session_state.page = "input"
