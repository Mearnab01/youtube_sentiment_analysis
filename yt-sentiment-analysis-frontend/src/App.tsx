// App.tsx
import { useEffect } from "react";
import { Activity, Zap, Users, MessageSquare, BarChart2, Hash, AlertTriangle, Loader2, Youtube, RefreshCw } from "lucide-react";
import { useYouTubeAnalyzer, type Metrics, type PredictionItem, type AppStatus } from "./hooks/useYouTubeAnalyzer";

// ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
// bg-[#000000]  surface-[#111111]  border-[#222222]
// accent: #2563EB (electric blue)
// positive: #22c55e  neutral: #6b7280  sarcastic/negative: #a855f7

// ─── HEADER ───────────────────────────────────────────────────────────────────
function Header({ status }: { status: AppStatus }) {
  const isConnected = status !== "error";

  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-[#222222] bg-[#000000]">
      <div className="flex items-center gap-2">
        <div className="w-5 h-5 flex items-center justify-center">
          <Youtube size={16} className="text-[#2563EB]" strokeWidth={2.5} />
        </div>
        <span
          className="text-[13px] font-bold tracking-[0.12em] uppercase text-white"
          style={{ fontFamily: "'JetBrains Mono', 'Courier New', monospace" }}
        >
          YT Insights
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span
          className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-[#22c55e]" : "bg-red-500"}`}
          style={{
            boxShadow: isConnected ? "0 0 6px #22c55e" : "0 0 6px #ef4444",
          }}
        />
        <span
          className="text-[10px] text-[#555555] font-mono tracking-wide"
        >
          23.20.221.231
        </span>
      </div>
    </header>
  );
}

// ─── METRIC GRID ──────────────────────────────────────────────────────────────
interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  accent?: boolean;
}

function MetricCard({ icon, label, value, accent }: MetricCardProps) {
  return (
    <div
      className="bg-[#111111] border border-[#222222] p-3 flex flex-col gap-2"
      style={{ borderRadius: 0 }}
    >
      <div className="flex items-center justify-between">
        <span
          className="text-[10px] uppercase tracking-[0.14em] text-[#555555]"
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
        >
          {label}
        </span>
        <span className={accent ? "text-[#2563EB]" : "text-[#333333]"}>{icon}</span>
      </div>
      <span
        className={`text-[22px] font-bold leading-none tracking-tight ${accent ? "text-[#2563EB]" : "text-white"}`}
        style={{ fontFamily: "'JetBrains Mono', monospace" }}
      >
        {value}
      </span>
    </div>
  );
}

function MetricGrid({ metrics }: { metrics: Metrics }) {
  return (
    <section className="px-4 pt-4">
      <SectionLabel icon={<BarChart2 size={11} />} text="Analysis Summary" />
      <div className="grid grid-cols-2 gap-px bg-[#222222] border border-[#222222]">
        <MetricCard
          icon={<MessageSquare size={11} />}
          label="Comments"
          value={metrics.totalComments.toLocaleString()}
        />
        <MetricCard
          icon={<Users size={11} />}
          label="Commenters"
          value={metrics.uniqueCommenters.toLocaleString()}
        />
        <MetricCard
          icon={<Hash size={11} />}
          label="Avg Length"
          value={`${metrics.avgWordLength}w`}
        />
        <MetricCard
          icon={<Activity size={11} />}
          label="Sentiment /10"
          value={metrics.normalizedSentimentScore}
          accent
        />
      </div>
    </section>
  );
}

// ─── VISUALIZATION GALLERY ────────────────────────────────────────────────────
function VisualizationGallery({
  chartUrl,
  trendUrl,
  wordcloudUrl,
}: {
  chartUrl: string | null;
  trendUrl: string | null;
  wordcloudUrl: string | null;
}) {
  const visuals: { label: string; url: string | null }[] = [
    { label: "Sentiment Distribution", url: chartUrl },
    { label: "Trend Over Time", url: trendUrl },
    { label: "Comment Wordcloud", url: wordcloudUrl },
  ];

  return (
    <section className="px-4 pt-4">
      <SectionLabel icon={<Zap size={11} />} text="Visualizations" />
      <div className="flex flex-col gap-px bg-[#222222] border border-[#222222]">
        {visuals.map(({ label, url }) =>
          url ? (
            <div key={label} className="bg-[#111111]">
              <div className="px-3 py-1.5 border-b border-[#1a1a1a]">
                <span
                  className="text-[10px] uppercase tracking-[0.14em] text-[#444444]"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  {label}
                </span>
              </div>
              <img
                src={url}
                alt={label}
                className="w-full block"
                style={{ display: "block" }}
              />
            </div>
          ) : null
        )}
      </div>
    </section>
  );
}

// ─── SENTIMENT BADGE ─────────────────────────────────────────────────────────
const SENTIMENT_MAP: Record<string, { label: string; color: string; bg: string }> = {
  "1":  { label: "Positive",  color: "#22c55e", bg: "rgba(34,197,94,0.08)"  },
  "0":  { label: "Neutral",   color: "#6b7280", bg: "rgba(107,114,128,0.08)" },
  "-1": { label: "Negative",  color: "#a855f7", bg: "rgba(168,85,247,0.08)" },
};

function SentimentBadge({ value }: { value: string }) {
  const s = SENTIMENT_MAP[value] ?? SENTIMENT_MAP["0"];
  return (
    <span
      className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5"
      style={{
        color: s.color,
        background: s.bg,
        border: `1px solid ${s.color}22`,
        fontFamily: "'JetBrains Mono', monospace",
        borderRadius: 0,
      }}
    >
      {s.label}
    </span>
  );
}

// ─── TOP COMMENTS ─────────────────────────────────────────────────────────────
function TopComments({ predictions }: { predictions: PredictionItem[] }) {
  const top25 = predictions.slice(0, 25);

  return (
    <section className="px-4 pt-4 pb-4">
      <SectionLabel icon={<MessageSquare size={11} />} text="Top 25 Comments" />
      <div className="border border-[#222222] bg-[#111111] overflow-y-auto" style={{ maxHeight: "240px" }}>
        {top25.map((item, i) => (
          <div
            key={i}
            className="px-3 py-2.5 border-b border-[#1a1a1a] last:border-b-0 hover:bg-[#161616] transition-colors"
          >
            <div className="flex items-start justify-between gap-2 mb-1">
              <span
                className="text-[10px] text-[#333333] font-mono leading-none mt-0.5"
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <SentimentBadge value={item.sentiment} />
            </div>
            <p
              className="text-[11px] text-[#aaaaaa] leading-relaxed"
              style={{ fontFamily: "system-ui, sans-serif" }}
            >
              {item.comment}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─── SECTION LABEL ────────────────────────────────────────────────────────────
function SectionLabel({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-1.5 mb-2">
      <span className="text-[#2563EB]">{icon}</span>
      <span
        className="text-[10px] uppercase tracking-[0.16em] text-[#444444]"
        style={{ fontFamily: "'JetBrains Mono', monospace" }}
      >
        {text}
      </span>
      <div className="flex-1 h-px bg-[#1a1a1a]" />
    </div>
  );
}

// ─── STATUS SCREEN ────────────────────────────────────────────────────────────
function StatusScreen({
  status,
  error,
  onRetry,
}: {
  status: AppStatus;
  error: string | null;
  onRetry: () => void;
}) {
  if (status === "not_youtube") {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-10 px-6 text-center">
        <Youtube size={28} className="text-[#222222]" />
        <p className="text-[11px] text-[#444444] font-mono tracking-wide uppercase">
          Navigate to a YouTube video
        </p>
        <p className="text-[10px] text-[#333333]">
          Open any youtube.com/watch?v=… page and re-open this popup.
        </p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-10 px-6 text-center">
        <AlertTriangle size={24} className="text-red-500" />
        <p className="text-[11px] text-red-400 font-mono tracking-wide uppercase">
          Error
        </p>
        <p className="text-[10px] text-[#555555] leading-relaxed">{error}</p>
        <button
          onClick={onRetry}
          className="flex items-center gap-1.5 mt-1 px-3 py-1.5 bg-[#111111] border border-[#333333] text-[10px] text-[#888888] uppercase tracking-widest font-mono hover:border-[#2563EB] hover:text-white transition-colors"
          style={{ borderRadius: 0 }}
        >
          <RefreshCw size={10} />
          Retry
        </button>
      </div>
    );
  }

  const STEP_MAP: Partial<Record<AppStatus, string>> = {
    idle: "Initializing…",
    fetching_comments: "Fetching comments…",
    analyzing: "Analyzing sentiment…",
  };

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 px-6">
      <Loader2 size={20} className="text-[#2563EB] animate-spin" />
      <p
        className="text-[11px] text-[#555555] font-mono tracking-widest uppercase"
      >
        {STEP_MAP[status] ?? "Processing…"}
      </p>
      {status === "fetching_comments" && (
        <div className="w-32 h-px bg-[#1a1a1a] relative overflow-hidden">
          <div
            className="absolute inset-0 h-full bg-[#2563EB]"
            style={{ animation: "slide 1.4s ease-in-out infinite" }}
          />
        </div>
      )}
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const { state, analyze } = useYouTubeAnalyzer();
  const { status, error, metrics, predictions, chartUrl, trendUrl, wordcloudUrl } = state;

  // Kick off analysis as soon as the popup opens
  useEffect(() => {
    analyze();
  }, [analyze]);

  const isDone = status === "done";

  return (
    <div
      className="bg-[#000000] min-h-full"
      style={{
        width: "360px",
        fontFamily: "system-ui, -apple-system, sans-serif",
        // Inject keyframe via inline style tag approach
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
        @keyframes slide {
          0%   { transform: translateX(-100%); }
          50%  { transform: translateX(0%); }
          100% { transform: translateX(100%); }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: #111111; }
        ::-webkit-scrollbar-thumb { background: #2563EB; }
      `}</style>

      <Header status={status} />

      {!isDone ? (
        <StatusScreen status={status} error={error} onRetry={analyze} />
      ) : (
        <div>
          {metrics && <MetricGrid metrics={metrics} />}

          <VisualizationGallery
            chartUrl={chartUrl}
            trendUrl={trendUrl}
            wordcloudUrl={wordcloudUrl}
          />

          {predictions.length > 0 && <TopComments predictions={predictions} />}

          {/* Re-analyze button */}
          <div className="px-4 pb-4">
            <button
              onClick={analyze}
              className="w-full flex items-center justify-center gap-2 py-2 bg-[#111111] border border-[#222222] text-[10px] text-[#555555] uppercase tracking-widest font-mono hover:border-[#2563EB] hover:text-[#2563EB] transition-colors"
              style={{ borderRadius: 0 }}
            >
              <RefreshCw size={10} />
              Re-analyze
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
