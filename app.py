import io
import re
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


logger = get_logger('Sentiment Analysis API')
wordnet.ensure_loaded()
logger.info("WordNet corpus pre-loaded at startup.")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "lgbm_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
EXPERIMENT_INFO_PATH = ROOT_DIR / "artifacts" / "experiment_info.json"
 
MAX_BATCH_SIZE = 500       
MAX_COMMENT_LENGTH = 5000  
SERVER_START_TIME = time.time()


app = Flask(__name__)
CORS(app)

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
 
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
 
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]
 
        fig = plt.figure(figsize=(12, 6))
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=SENTIMENT_LABELS[sentiment_value],
                color=SENTIMENT_COLORS[sentiment_value]
            )
 
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
 
        plt.legend()
        plt.tight_layout()
 
        return figure_to_png_response(fig)
    except Exception as e:
        logger.error("Error in /generate_trend_graph: %s", e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
 