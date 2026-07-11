import streamlit as st
import pickle
import re
import requests
import pandas as pd
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="📊", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
ASSETS_DIR = ROOT_DIR / "assets"

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
LEMMATIZER = WordNetLemmatizer()
_CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s!?.,]')


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def preprocess_comment(comment: str) -> str:
    comment = comment.lower().strip()
    comment = _CLEAN_PATTERN.sub('', comment)
    comment = ' '.join(w for w in comment.split() if w not in STOP_WORDS)
    comment = ' '.join(LEMMATIZER.lemmatize(w) for w in comment.split())
    return comment


def fetch_comments(video_id: str, api_key: str, max_comments: int = 500):
    comments = []
    page_token = ""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    while len(comments) < max_comments:
        params = {
            "part": "snippet", "videoId": video_id,
            "maxResults": 100, "pageToken": page_token, "key": api_key
        }
        resp = requests.get(url, params=params).json()
        if "items" not in resp:
            break
        for item in resp["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            timestamp = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
            if text.strip():
                comments.append({"text": text, "timestamp": timestamp})
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comments


# ==========================================================
# UI
# ==========================================================
st.title("📊 YouTube Comment Sentiment Analyzer")
st.caption(
    "This is a standalone demo of the model that powers my Chrome extension "
    "(charts, wordcloud, AI insights). Try it below with any video, then "
    "scroll down to see actual screenshots of the extension in action."
)

video_id = st.text_input("YouTube Video ID (e.g. dQw4w9WgXcQ)")
api_key = 'AIzaSyDAdRpmLSD0SdaX_1shixX5TGoN-qTftIM'

if st.button("Analyze") and video_id and api_key:
    model, vectorizer = load_model()

    with st.spinner("Fetching comments..."):
        comments = fetch_comments(video_id, api_key)

    if not comments:
        st.warning("No comments found, or invalid video ID / API key.")
    else:
        with st.spinner(f"Analyzing {len(comments)} comments..."):
            texts = [c["text"] for c in comments]
            preprocessed = [preprocess_comment(c) for c in texts]
            transformed = vectorizer.transform(preprocessed)
            predictions = model.predict(transformed.toarray())

        pos = int(sum(1 for p in predictions if p == 1))
        neu = int(sum(1 for p in predictions if p == 0))
        neg = int(sum(1 for p in predictions if p == -1))
        total = len(comments)

        st.subheader("Sentiment Distribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Positive", pos, f"{pos/total*100:.1f}%")
        col2.metric("⚪ Neutral", neu, f"{neu/total*100:.1f}%")
        col3.metric("🔴 Negative", neg, f"{neg/total*100:.1f}%")

        st.bar_chart(pd.DataFrame({"count": [pos, neu, neg]}, index=["Positive", "Neutral", "Negative"]))

        st.subheader("Sentiment Trend Over Time")
        df = pd.DataFrame({"timestamp": [c["timestamp"] for c in comments], "sentiment": predictions})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        trend = df.resample("W")["sentiment"].value_counts().unstack(fill_value=0)
        st.line_chart(trend)

        st.subheader("Most Common Words")
        all_words = " ".join(preprocessed).split()
        common = Counter(all_words).most_common(20)
        st.dataframe(pd.DataFrame(common, columns=["word", "count"]), use_container_width=True)

        st.subheader("Sample Comments")
        for c, p in list(zip(texts, predictions))[:15]:
            label = {1: "🟢 Positive", 0: "⚪ Neutral", -1: "🔴 Negative"}[p]
            st.write(f"**{label}**: {c}")

st.divider()

# ==========================================================
# Screenshot gallery — actual Chrome extension in action
# ==========================================================
st.header("📸 Chrome Extension — Actual Screenshots")
st.caption(
    "This Streamlit page demos the underlying model. The real product is a "
    "Chrome extension with a richer UI (side panel, wordcloud, AI-generated "
    "insights). Screenshots below are from the live extension."
)

# TODO: drop your screenshots into an `assets/` folder next to this file,
# named image1.png, image2.png, etc. (or rename the list below to match
# whatever filenames you actually use).
screenshot_files = sorted(ASSETS_DIR.glob("image*.png")) if ASSETS_DIR.exists() else []

if screenshot_files:
    captions = [
        "Sentiment distribution + comment summary",
        "Sentiment trend over time",
        "Comment wordcloud",
        "AI-generated insight summary",
        "Top comments ranked by sentiment",
    ]
    for i, img_path in enumerate(screenshot_files):
        caption = captions[i] if i < len(captions) else img_path.stem
        st.image(str(img_path), caption=caption, use_container_width=True)
else:
    st.info(
        "No screenshots found yet. Add PNG files to an `assets/` folder "
        "next to this script, named image1.png, image2.png, etc."
    )