// useYouTubeAnalyzer.ts
import { useState, useCallback } from "react";

const YT_API_KEY = "AIzaSyAHxO8ZCKPLODbhgSQcDV49Bv8cgkOA8Z4";
const BACKEND_URL = "http://23.20.221.231:8080";

export type AppStatus =
  | "idle"
  | "fetching_comments"
  | "analyzing"
  | "done"
  | "error"
  | "not_youtube";

export interface Comment {
  text: string;
  timestamp: string;
  authorId: string;
}

export interface PredictionItem {
  comment: string;
  sentiment: string; // "1", "0", "-1"
  timestamp: string;
}

export interface Metrics {
  totalComments: number;
  uniqueCommenters: number;
  avgWordLength: string;
  normalizedSentimentScore: string;
}

export interface AnalyzerState {
  status: AppStatus;
  error: string | null;
  videoId: string | null;
  metrics: Metrics | null;
  predictions: PredictionItem[];
  chartUrl: string | null;
  trendUrl: string | null;
  wordcloudUrl: string | null;
}

export function useYouTubeAnalyzer() {
  const [state, setState] = useState<AnalyzerState>({
    status: "idle",
    error: null,
    videoId: null,
    metrics: null,
    predictions: [],
    chartUrl: null,
    trendUrl: null,
    wordcloudUrl: null,
  });

  const setPartial = (partial: Partial<AnalyzerState>) =>
    setState((prev) => ({ ...prev, ...partial }));

  // ─── YouTube Comment Fetcher ───────────────────────────────────────────────
  async function fetchComments(videoId: string): Promise<Comment[]> {
    const comments: Comment[] = [];
    let pageToken = "";

    while (comments.length < 500) {
      const url = new URL(
        "https://www.googleapis.com/youtube/v3/commentThreads"
      );
      url.searchParams.set("part", "snippet");
      url.searchParams.set("videoId", videoId);
      url.searchParams.set("maxResults", "100");
      url.searchParams.set("key", YT_API_KEY);
      if (pageToken) url.searchParams.set("pageToken", pageToken);

      const res = await fetch(url.toString());
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(
          err?.error?.message || `YouTube API error: ${res.status}`
        );
      }

      const data = await res.json();
      if (data.items) {
        for (const item of data.items) {
          const snip = item.snippet.topLevelComment.snippet;
          comments.push({
            text: snip.textOriginal,
            timestamp: snip.publishedAt,
            authorId: snip.authorChannelId?.value ?? "unknown",
          });
        }
      }

      pageToken = data.nextPageToken ?? "";
      if (!pageToken) break;
    }

    return comments;
  }

  // ─── Backend Calls ─────────────────────────────────────────────────────────
  async function getSentimentPredictions(
    comments: Comment[]
  ): Promise<PredictionItem[]> {
    const res = await fetch(`${BACKEND_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });
    if (!res.ok) throw new Error(`Prediction API error: ${res.status}`);
    return res.json();
  }

  async function fetchImageBlob(endpoint: string, body: object): Promise<string> {
    const res = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${endpoint} failed: ${res.status}`);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  // ─── Main Orchestrator ─────────────────────────────────────────────────────
  const analyze = useCallback(async () => {
    setState({
      status: "fetching_comments",
      error: null,
      videoId: null,
      metrics: null,
      predictions: [],
      chartUrl: null,
      trendUrl: null,
      wordcloudUrl: null,
    });

    try {
      // 1. Get current tab URL
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      const url = tabs[0]?.url ?? "";
      const match = url.match(/^https?:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);

      if (!match) {
        setPartial({ status: "not_youtube" });
        return;
      }

      const videoId = match[1];
      setPartial({ videoId });

      // 2. Fetch comments
      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        setPartial({ status: "error", error: "No comments found for this video." });
        return;
      }

      setPartial({ status: "analyzing" });

      // 3. Sentiment predictions
      const predictions = await getSentimentPredictions(comments);

      // 4. Compute metrics
      const sentimentCounts: Record<string, number> = { "1": 0, "0": 0, "-1": 0 };
      const sentimentData: { timestamp: string; sentiment: number }[] = [];
      let totalSentiment = 0;

      for (const item of predictions) {
        const s = parseInt(item.sentiment, 10);
        sentimentCounts[item.sentiment] = (sentimentCounts[item.sentiment] ?? 0) + 1;
        sentimentData.push({ timestamp: item.timestamp, sentiment: s });
        totalSentiment += s;
      }

      const totalComments = comments.length;
      const uniqueCommenters = new Set(comments.map((c) => c.authorId)).size;
      const totalWords = comments.reduce(
        (sum, c) => sum + c.text.split(/\s+/).filter((w) => w.length > 0).length,
        0
      );
      const avgWordLength = (totalWords / totalComments).toFixed(1);
      const avgSentiment = totalSentiment / totalComments;
      const normalizedSentimentScore = (((avgSentiment + 1) / 2) * 10).toFixed(1);

      const metrics: Metrics = {
        totalComments,
        uniqueCommenters,
        avgWordLength,
        normalizedSentimentScore,
      };

      // 5. Fetch visualizations in parallel
      const [chartUrl, trendUrl, wordcloudUrl] = await Promise.all([
        fetchImageBlob("/generate_chart", { sentiment_counts: sentimentCounts }),
        fetchImageBlob("/generate_trend_graph", { sentiment_data: sentimentData }),
        fetchImageBlob("/generate_wordcloud", {
          comments: comments.map((c) => c.text),
        }),
      ]);

      setPartial({
        status: "done",
        metrics,
        predictions,
        chartUrl,
        trendUrl,
        wordcloudUrl,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "An unknown error occurred.";
      setPartial({ status: "error", error: msg });
    }
  }, []);

  return { state, analyze };
}
