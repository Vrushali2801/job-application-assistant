# Job Application Assistant

Upload your CV (PDF) and paste a job description. The app analyses your fit, researches the company, identifies skill gaps with project suggestions to close them, and rewrites only the CV bullets that genuinely need improving — with a side-by-side before/after view you can copy directly.

## Features

- **Fit score** — 0–100 match score with strong matches and skill gaps highlighted
- **Company overview** — auto-researched from the web
- **Gap suggestions** — concrete projects you could build to demonstrate missing skills (no filler advice)
- **Selective CV rewrites** — only bullets that genuinely benefit from a change are rewritten; keywords are never forced in if they don't fit the candidate's background
- **Copy to clipboard** — copy individual rewrites or all selected at once

## How to run locally

```bash
# 1. Add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 2. Install dependencies
uv sync

# 3. Start the server
uv run uvicorn backend.main:app --reload
```

Then open `http://localhost:8000` in your browser.

## What you need

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- A free [Groq API key](https://console.groq.com)

## Project structure

```
├── backend/
│   ├── main.py      # FastAPI app — /analyse endpoint + static frontend serving
│   ├── agent.py     # LangGraph agent — 5-node pipeline
│   ├── parser.py    # PDF text extraction (PyMuPDF)
│   ├── tools.py     # DuckDuckGo web search tool
│   └── __init__.py
├── frontend/
│   └── index.html   # Single-page UI (Tailwind CSS, no build step)
├── main.py          # Dev entry point
├── Procfile         # Railway deployment
├── pyproject.toml
└── requirements.txt
```

## Agent pipeline

```
extract_info → research_company → analyse_fit → suggest_gaps → rewrite_bullets
```

1. **extract_info** — pulls candidate details, required skills, and CV bullets from the uploaded PDF and job description
2. **research_company** — searches the web for a company overview
3. **analyse_fit** — scores the match, identifies strong matches, skill gaps, and missing keywords
4. **suggest_gaps** — generates specific project ideas for each skill gap relevant to the candidate's background
5. **rewrite_bullets** — rewrites only bullets that genuinely benefit from a change; never invents experience or forces irrelevant keywords

## Tech stack

- **Backend** — FastAPI, LangGraph, LangChain
- **LLM** — Llama 3.3 70B via Groq
- **PDF parsing** — PyMuPDF
- **Web search** — DuckDuckGo (via `duckduckgo-search`)
- **Frontend** — Vanilla JS, Tailwind CSS (CDN)
