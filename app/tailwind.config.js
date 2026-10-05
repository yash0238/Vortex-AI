/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "primary": "#2563eb",
        "secondary": "#0ea5e1",
        "dark-bg": "#0f172a",
        "card-bg": "#1e293b",
        "border": "#334155",
        "crisis": "#ef4444",
        "normal": "#22c55e",
      },
    },
  },
  plugins: [],
}
