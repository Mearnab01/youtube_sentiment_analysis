import io
import re
import os
import json
import time
from pathlib import Path
from functools import wraps
 
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
 
import numpy as np
import pandas as pd
import pickle
 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
 
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from src.utils.logger import get_logger

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# configure paths and constants
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
EXPERIMENT_INFO_PATH = ROOT_DIR / "artifacts" / "experiment_info.json"
 
MAX_BATCH_SIZE = 2000
MAX_COMMENT_LENGTH = 10000  
SERVER_START_TIME = time.time()
 
GROQ_MODEL = "llama-3.3-70b-versatile"

logger = get_logger('Sentiment Analysis API')

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
wordnet.ensure_loaded()
logger.info("WordNet corpus pre-loaded at startup.")

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
LEMMATIZER = WordNetLemmatizer()
_CLEAN_PATTERN = re.compile(r'[^A-Za-z0-9\s!?.,]')
_NEWLINE_PATTERN = re.compile(r'\n')
 
SENTIMENT_LABELS = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
SENTIMENT_COLORS = {-1: 'red', 0: 'gray', 1: 'green'}


def preprocess_comment(comment:str)->str:
    try:
        comment = comment.lower().strip()
        comment = _NEWLINE_PATTERN.sub(' ', comment)
        comment = _CLEAN_PATTERN.sub('', comment)
        comment = ' '.join([word for word in comment.split() if word not in STOP_WORDS])
        comment = ' '.join([LEMMATIZER.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        raise
    
def load_model_and_vectorizer(model_path: Path, vectorizer_path: Path):
    """Load the trained model and vectorizer from local disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
 
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
 
    logger.info("Model and vectorizer loaded from %s, %s", model_path, vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)
SUPPORTS_PROBA = hasattr(model, "predict_proba")
 
 
def load_experiment_info() -> dict:
    """Load evaluation metadata (accuracy, params) produced by model_evaluation.py, if present."""
    if not EXPERIMENT_INFO_PATH.exists():
        return {}
    try:
        with open(EXPERIMENT_INFO_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read experiment_info.json: %s", e)
        return {}
 
 
def validate_comments(comments) -> tuple[bool, str]:
    """Validate a batch of comments before running inference. Returns (is_valid, error_message)."""
    if not isinstance(comments, list):
        return False, "'comments' must be a list"
    if len(comments) == 0:
        return False, "'comments' list is empty"
    if len(comments) > MAX_BATCH_SIZE:
        return False, f"Batch too large: {len(comments)} comments (max {MAX_BATCH_SIZE})"
    for c in comments:
        if not isinstance(c, str) or not c.strip():
            return False, "Each comment must be a non-empty string"
        if len(c) > MAX_COMMENT_LENGTH:
            return False, f"Comment exceeds max length of {MAX_COMMENT_LENGTH} characters"
    return True, ""
 
 
def timed_route(f):
    """Log request latency for each endpoint call."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        response = f(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        logger.info("%s completed in %.1fms", request.path, elapsed_ms)
        return response
    return wrapper
 
 
def predict_sentiments(comments: list[str]):
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    dense = transformed.toarray()
 
    predictions = model.predict(dense).tolist()
 
    if SUPPORTS_PROBA:
        proba = model.predict_proba(dense)
        confidences = [float(np.max(row)) for row in proba]
    else:
        confidences = [None] * len(predictions)
 
    return predictions, confidences
 
 
def figure_to_png_response(fig=None, transparent: bool = False):
    """Render the current (or given) matplotlib figure to a PNG HTTP response and close it."""
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', transparent=transparent)
    plt.close(fig if fig is not None else 'all')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')
 
 
# ==========================================================
# Routes
# ==========================================================
@app.route('/')
def home():
    return "Welcome to our flask api"
 
 
@app.route('/health', methods=['GET'])
def health():
    """Liveness + model metadata endpoint. Useful for uptime checks and debugging deployments."""
    info = load_experiment_info()
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "model_type": type(model).__name__,
        "supports_confidence": SUPPORTS_PROBA,
        "model_path": str(MODEL_PATH),
        "evaluation_metrics": {
            "accuracy": info.get("accuracy"),
            "macro_avg": info.get("macro_avg"),
        } if info else None,
    })
 
 
@app.route('/predict_with_timestamps', methods=['POST'])
@timed_route
def predict_with_timestamps():
    data = request.get_json(silent=True) or {}
    comments_data = data.get('comments')
 
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400
 
    try:
        raw_comments = [item['text'] for item in comments_data]
        raw_timestamps = [item['timestamp'] for item in comments_data]
    except (KeyError, TypeError):
        return jsonify({"error": "Each item must have 'text' and 'timestamp'"}), 400
    
    comments, timestamps = [], []
    for text, ts in zip(raw_comments, raw_timestamps):
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > MAX_COMMENT_LENGTH:
            text = text[:MAX_COMMENT_LENGTH]
        comments.append(text)
        timestamps.append(ts)
 
    if not comments:
        return jsonify({"error": "No valid (non-empty) comments after sanitizing"}), 400
    if len(comments) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch too large: {len(comments)} comments (max {MAX_BATCH_SIZE})"}), 400
 
    try:
        predictions, confidences = predict_sentiments(comments)
    except Exception as e:
        logger.error("Prediction with timestamps failed: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
 
    response = [
        {"comment": c, "sentiment": str(s), "confidence": conf, "timestamp": t}
        for c, s, conf, t in zip(comments, predictions, confidences, timestamps)
    ]
    return jsonify(response)
 
 
@app.route('/predict', methods=['POST'])
@timed_route
def predict():
    data = request.get_json(silent=True) or {}
    raw_comments = data.get('comments')
 
    if not raw_comments or not isinstance(raw_comments, list):
        return jsonify({"error": "'comments' must be a non-empty list"}), 400
 
    comments = []
    for text in raw_comments:
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > MAX_COMMENT_LENGTH:
            text = text[:MAX_COMMENT_LENGTH]
        comments.append(text)
 
    if not comments:
        return jsonify({"error": "No valid (non-empty) comments after sanitizing"}), 400
    if len(comments) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch too large: {len(comments)} comments (max {MAX_BATCH_SIZE})"}), 400
 
    try:
        predictions, confidences = predict_sentiments(comments)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
 
    response = [
        {"comment": c, "sentiment": s, "confidence": conf}
        for c, s, conf in zip(comments, predictions, confidences)
    ]
    return jsonify(response)
 
 
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json(silent=True) or {}
        sentiment_counts = data.get('sentiment_counts')
 
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400
 
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
 
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
 
        fig = plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')
 
        return figure_to_png_response(fig, transparent=True)
    except Exception as e:
        logger.error("Error in /generate_chart: %s", e)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500
 
 
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json(silent=True) or {}
        comments = data.get('comments')
 
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
 
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)
 
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=STOP_WORDS,
            collocations=False
        ).generate(text)
 
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
 
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.error("Error in /generate_wordcloud: %s", e)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
 
 
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json(silent=True) or {}
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        date_span_days = (df.index.max() - df.index.min()).days
        if date_span_days <= 45:
            freq, date_fmt = 'W', '%b %d'
            locator = mdates.WeekdayLocator(interval=1)
        elif date_span_days <= 365:
            freq, date_fmt = 'ME', '%b %Y'
            locator = mdates.MonthLocator(interval=1)
        else:
            freq, date_fmt = 'QE', '%b %Y'
            locator = mdates.MonthLocator(interval=3)

        monthly_counts = df.resample(freq)['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        BG_COLOR = '#1a1d24'
        GRID_COLOR = '#3a3f4b'
        TEXT_COLOR = '#e8e8ea'

        COLORS = {
            -1: '#ff6384',  # softer red
            0: '#9aa0ac',   # lighter gray
            1: '#4ade80',   # softer green
        }

        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        for sentiment_value in [-1, 0, 1]:
            ax.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                markersize=7,
                linewidth=2.5,
                label=SENTIMENT_LABELS[sentiment_value],
                color=COLORS[sentiment_value]
            )

        ax.set_title('Sentiment Over Time', fontsize=16, fontweight='bold', color=TEXT_COLOR, pad=14)
        ax.set_xlabel('')  # axis label redundant with date-formatted ticks
        ax.set_ylabel('% of Comments', fontsize=12, color=TEXT_COLOR)

        ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(GRID_COLOR)
        ax.spines['bottom'].set_color(GRID_COLOR)

        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        legend = ax.legend(
            loc='upper left', frameon=False, fontsize=11,
            labelcolor=TEXT_COLOR
        )

        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', facecolor=BG_COLOR, dpi=150)
        plt.close(fig)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.error("Error in /generate_trend_graph: %s", e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/generate_insights', methods=['POST'])  
@timed_route
def generate_insights():
    try:
        data = request.get_json(silent=True) or {}
        predictions = data.get('predictions')  
        sentiment_counts = data.get('sentiment_counts')
 
        if not predictions or not sentiment_counts:
            return jsonify({"error": "predictions and sentiment_counts are required"}), 400
 
        negative_comments = [p for p in predictions if str(p.get('sentiment')) == '-1']
        top_negative = max(negative_comments, key=lambda c: c.get('likeCount', 0), default=None)
 
        sample_positive = [p['comment'] for p in predictions if str(p.get('sentiment')) == '1'][:5]
        sample_negative = [p['comment'] for p in predictions if str(p.get('sentiment')) == '-1'][:5]
 
        total = sum(sentiment_counts.values())
        pos_pct = round(sentiment_counts.get('1', 0) / total * 100, 1)
        neu_pct = round(sentiment_counts.get('0', 0) / total * 100, 1)
        neg_pct = round(sentiment_counts.get('-1', 0) / total * 100, 1)
 
        prompt = f"""You are analyzing YouTube comment sentiment for a content creator.
 
            Sentiment breakdown: {pos_pct}% positive, {neu_pct}% neutral, {neg_pct}% negative.
            
            Sample positive comments: {sample_positive}
            Sample negative comments: {sample_negative}
            
            Write a 2-3 sentence summary a creator would actually find useful — call out the
            overall tone AND any specific recurring complaint or praise you notice in the
            samples. Be direct and concrete, not generic. No preamble, just the summary."""
            
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        summary = completion.choices[0].message.content.strip()
 
        return jsonify({
            "summary": summary,
            "top_negative_comment": top_negative
        })
 
    except Exception as e:
        logger.error("Error in /generate_insights: %s", e)
        return jsonify({"error": f"Insight generation failed: {str(e)}"}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
 