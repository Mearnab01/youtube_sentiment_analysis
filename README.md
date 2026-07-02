# YT Insights — YouTube Comment Sentiment Analysis (MLOps Pipeline + Chrome Extension)

> Paste a YouTube video → get sentiment breakdown, trend graph, and wordcloud of the comments — powered by a versioned, tracked ML pipeline and shipped as a real Chrome extension.

---

## What it does

YT Insights pulls up to 500 comments from any YouTube video via the YouTube Data API, runs them through a trained LightGBM sentiment classifier, and renders the results (positive/neutral/negative split, a sentiment-over-time trend graph, and a wordcloud) directly inside a Chrome extension popup.

This isn't just a notebook model wrapped in an API — it's a full **DVC-versioned, MLflow-tracked ML pipeline** feeding a **Flask inference service**, consumed by a **React + TypeScript + Tailwind Manifest V3 Chrome extension**.

---

## Architecture

```
Chrome Extension (React + TS)
        │
        │ 1. YouTube Data API — fetch up to 500 comments
        ▼
Flask Inference Server (EC2, :8080)
        │
        ├──► /predict_with_timestamps   → TF-IDF vectorizer → LightGBM model → sentiment labels
        ├──► /generate_chart            → sentiment distribution pie chart (PNG)
        ├──► /generate_trend_graph      → sentiment-over-time line chart (PNG)
        └──► /generate_wordcloud        → wordcloud image (PNG)
```

**Training pipeline (DVC-orchestrated, 4 stages):**

```
data_ingestion  →  data_preprocessing  →  model_building  →  model_evaluation
  (raw split)       (clean + dedup)      (TF-IDF + LightGBM)    (metrics → MLflow)
```

Every stage is a DVC pipeline stage with declared deps, params, and outputs (`dvc.yaml`) — so `dvc repro` rebuilds only the stages whose inputs actually changed, and every run is reproducible from a params.yaml diff.

---

## Experimentation trail

The model wasn't the first thing that worked — it's the result of 7 tracked experiments, each in its own notebook:

| # | Experiment | What was tested |
|---|---|---|
| 1 | Baseline model | First working sentiment classifier, no tuning |
| 2 | BoW vs TF-IDF | Compared bag-of-words against TF-IDF features |
| 3 | TF-IDF (1,3)-gram, max_features sweep | Trigram range + vocabulary size tuning |
| 4 | Imbalanced data handling | Class weighting / resampling for skewed sentiment distribution |
| 5 | XGBoost + hyperparameter tuning | XGBoost as a candidate model |
| 6 | LightGBM detailed HPT | Final model family — deeper hyperparameter search |

**Final config (`params.yaml`):** TF-IDF with `ngram_range=(1,3)`, `max_features=10000`, feeding a LightGBM classifier (`learning_rate=0.034`, `num_leaves=140`, `n_estimators=449`, `reg_alpha=1.95`, `reg_lambda=0.002`) — tuned via the HPT sweep in experiment 6.

---

## Why this stack

**Why DVC over just running scripts manually?**
Each pipeline stage declares its exact dependencies and parameters. Change `max_features` in `params.yaml` and `dvc repro` reruns model_building and model_evaluation only — data_ingestion is untouched since its deps didn't change. That's the difference between "I think I retrained on the right data" and knowing it.

**Why LightGBM over XGBoost?**
Both were tuned (experiments 5 and 6) — LightGBM's leaf-wise growth handled this feature space better after tuning and trains faster on the TF-IDF sparse matrix, which mattered for iterating quickly through the HPT sweep.

**Why MLflow?**
`model_evaluation.py` logs metrics to `experiment_info.json` and MLflow — so every experiment's metrics are comparable side by side instead of living in scattered notebook print statements.

**Why a Chrome extension instead of a web app?**
Sentiment analysis on YouTube comments is only useful *in context* — while you're watching the video. A popup that reads the current tab's video ID and overlays results is a better UX fit than a separate site you'd have to copy-paste a URL into.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | LightGBM (tuned via HPT), TF-IDF vectorizer |
| Pipeline orchestration | DVC (4-stage pipeline: ingest → preprocess → train → evaluate) |
| Experiment tracking | MLflow |
| Backend | Flask, Flask-CORS, served on EC2 |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Extension | Chrome Manifest V3 |
| Data source | YouTube Data API v3 (`commentThreads`) |
| Storage | boto3 (S3 for DVC remote) |
| Containerization | Docker |
| NLP utils | NLTK, wordcloud |

---

## Project Structure

```
youtube_sentiment_analysis/
├── src/
│   ├── model/
│   │   ├── model_building.py       # TF-IDF + LightGBM training
│   │   ├── model_evaluation.py     # metrics → experiment_info.json + MLflow
│   │   └── register_model.py       # MLflow model registry push
│   └── utils/
│       └── logger.py
├── flask/
│   └── app.py                      # inference server — predict + chart/trend/wordcloud endpoints
├── notebooks/                      # 7 experiments, baseline → tuned LightGBM
├── yt-sentiment-analysis-frontend/ # Chrome extension (React + TS + Tailwind)
│   ├── manifest.json
│   ├── popup.html
│   └── src/
│       ├── App.tsx                 # UI: header, metric grid, charts, top comments
│       └── hooks/useYouTubeAnalyzer.ts   # fetch + state machine
├── dvc.yaml                        # 4-stage pipeline definition
├── params.yaml                     # single source of truth for all hyperparameters
├── Dockerfile
└── requirements.txt
```

---

## Chrome Extension — State Machine

```
idle → fetching_comments → analyzing → done
                                     ↘ error
                        ↘ not_youtube
```

## Design tokens (extension UI)

| Token | Value | Usage |
|---|---|---|
| Background | `#000000` | Page background |
| Accent | `#2563EB` | Primary actions |
| Positive | `#22c55e` | Positive sentiment |
| Neutral | `#6b7280` | Neutral sentiment |
| Negative | `#a855f7` | Negative/sarcastic sentiment |

Industrial-minimalist spec — zero border radius across all components.

---

## Setup

### Backend

```bash
git clone https://github.com/Mearnab01/youtube_sentiment_analysis.git
cd youtube_sentiment_analysis
pip install -r requirements.txt

# Reproduce the full training pipeline
dvc repro

# Run inference server
python flask/app.py
```

### Chrome Extension

```bash
cd yt-sentiment-analysis-frontend
npm install
npm run build
```

Then: `chrome://extensions` → enable Developer mode → **Load unpacked** → select `/dist`.

### Docker (inference server)

```bash
docker build -t yt-insights-backend .
docker run -p 8080:8080 yt-insights-backend
```

---

## API

```
POST /predict_with_timestamps   Comments (+ timestamps) → sentiment labels
POST /generate_chart            Sentiment distribution → pie chart PNG
POST /generate_trend_graph      Sentiment over time → line chart PNG
POST /generate_wordcloud        Comment text → wordcloud PNG
```

---

## Author

**Arnab Nath** — [GitHub](https://github.com/Mearnab01)

---
