# Job Application Assistant

Upload your CV (PDF) and paste a job description (or drop in a URL). The app analyses your fit, researches the company, rewrites only the bullets that genuinely need improving, generates a personalised cover letter, and lets you edit and download a tailored resume — all in under a minute.

## Features

- **Fit score** — 0–100 match score with strong matches and skill gaps highlighted
- **ATS keyword analysis** — top 8–10 keywords from the job description rated Strong / Weak / Absent in your CV, with required vs preferred weight
- **Company overview** — auto-researched from the web
- **Gap suggestions** — concrete project ideas you could build to demonstrate missing skills (no filler advice, no "take a course")
- **Selective CV rewrites** — only bullets that genuinely benefit from a change are rewritten; keywords are never forced in if they don't fit the candidate's background
- **Cover letter** — 3-paragraph letter grounded in your actual CV, referencing specific details from the job posting
- **Resume editor + PDF download** — live in-browser editor with rewrites already spliced in; download as a clean, professional PDF

## How to run locally

```bash
# 1. Copy the example env file and add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 2. Install dependencies
uv sync

# 3. Install the Playwright browser (needed for URL scraping and PDF export)
uv run playwright install chromium

# 4. Start the server
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
│   ├── main.py      # FastAPI app — streaming /analyse endpoint, resume editor, PDF export
│   ├── agent.py     # LangGraph pipeline — 6-node graph
│   ├── parser.py    # PDF text extraction (PyMuPDF)
│   ├── tools.py     # DuckDuckGo web search
│   └── __init__.py
├── frontend/
│   └── index.html   # Landing page + single-page app (vanilla JS, custom CSS)
├── main.py          # Dev entry point (uvicorn wrapper)
├── Procfile         # Railway deployment
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Agent pipeline

Nodes in the same column run in parallel:

```
extract_info
     │
     ├── research_company ──────────────────┐
     │                                      │
     └── analyse_fit                        │
              │                             │
              ├── suggest_gaps ─────────────┤
              │                             │
              └── rewrite_bullets ──────────┤
                                            │
                                     generate_cover_letter
```

1. **extract_info** — pulls candidate details, required skills, and all CV bullets from the PDF and job description
2. **research_company** — searches the web for a company overview *(parallel with analyse_fit)*
3. **analyse_fit** — scores the match, identifies strong matches, skill gaps, missing keywords, and rates ATS keywords *(parallel with research_company)*
4. **suggest_gaps** — generates specific project ideas for each skill gap, relevant to the candidate's existing background *(parallel with rewrite_bullets)*
5. **rewrite_bullets** — rewrites only bullets that genuinely benefit from a change; never invents experience or forces irrelevant keywords *(parallel with suggest_gaps)*
6. **generate_cover_letter** — writes a 3-paragraph cover letter grounded in the candidate's actual CV achievements; runs after all five preceding nodes complete

Results stream to the browser as each node completes, so you see partial output immediately.

## Tech stack

- **Backend** — FastAPI, LangGraph, LangChain
- **LLM** — Llama 3.3 70B via Groq
- **PDF parsing** — PyMuPDF
- **Web scraping** — httpx + lxml (fast path), Playwright (fallback for JS-heavy pages)
- **Web search** — DuckDuckGo (via `duckduckgo-search`)
- **Frontend** — Vanilla JS, custom CSS (no build step)
