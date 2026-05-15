// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}", "./popup.html"],
  theme: {
    extend: {
      // No radius overrides needed — use rounded-none everywhere in components
      fontFamily: {
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      colors: {
        surface: "#111111",
        border: "#222222",
        accent: "#2563EB",
        positive: "#22c55e",
        neutral: "#6b7280",
        negative: "#a855f7",
      },
    },
  },
  plugins: [],
};
