# YT Insights — Chrome Extension v2

Industrial-minimalist YouTube comment sentiment analyser, rebuilt in React + TypeScript + Tailwind CSS.

---

## Project Structure

```
yt-insights-extension/
├── manifest.json              # Manifest V3
├── popup.html                 # Extension popup entry
├── vite.config.ts             # Vite + CRXJS bundler config
├── tailwind.config.js
├── tsconfig.json
├── postcss.config.js
├── package.json
└── src/
    ├── index.tsx              # React root
    ├── index.css              # Tailwind base
    ├── App.tsx                # All UI components
    └── hooks/
        └── useYouTubeAnalyzer.ts   # All API + state logic
```

---

## Setup & Build

```bash
# 1. Install dependencies
npm install

# 2. Build the extension
npm run build

# 3. Load in Chrome
#    → chrome://extensions → Enable "Developer mode"
#    → "Load unpacked" → select the /dist folder
```

---

## Components

| Component | Location | Responsibility |
|---|---|---|
| `Header` | `App.tsx` | Title + backend status dot |
| `MetricGrid` | `App.tsx` | 4-up metric cards |
| `VisualizationGallery` | `App.tsx` | Pie chart, trend graph, wordcloud |
| `TopComments` | `App.tsx` | Scrollable top-25 comments list |
| `StatusScreen` | `App.tsx` | Idle / Loading / Error / Not-YouTube states |
| `useYouTubeAnalyzer` | `hooks/` | All fetch logic + state machine |

---

## State Machine

```
idle → fetching_comments → analyzing → done
                                     ↘ error
                        ↘ not_youtube
```

---

## API Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `https://www.googleapis.com/youtube/v3/commentThreads` | GET | Fetch up to 500 comments |
| `http://23.20.221.231:8080/predict_with_timestamps` | POST | Sentiment predictions |
| `http://23.20.221.231:8080/generate_chart` | POST | Pie chart image |
| `http://23.20.221.231:8080/generate_trend_graph` | POST | Trend graph image |
| `http://23.20.221.231:8080/generate_wordcloud` | POST | Wordcloud image |

---

## Design Tokens

| Token | Value | Usage |
|---|---|---|
| Background | `#000000` | Page background |
| Surface | `#111111` | Cards, panels |
| Border | `#222222` | All borders |
| Accent (Electric Blue) | `#2563EB` | Primary actions, values |
| Positive | `#22c55e` | Positive sentiment |
| Neutral | `#6b7280` | Neutral sentiment |
| Negative | `#a855f7` | Negative/sarcastic sentiment |

All borders use `rounded-none` (zero radius) per the industrial-minimalist spec.

---

## Notes

- The YouTube Data API key is embedded in `useYouTubeAnalyzer.ts`. For production, store it in a `.env` file:
  ```
  VITE_YT_API_KEY=your_key_here
  ```
  Then reference it as `import.meta.env.VITE_YT_API_KEY`.

- Place your extension icons (16×16, 32×32, 48×48, 128×128 PNGs) in `public/icons/`.
