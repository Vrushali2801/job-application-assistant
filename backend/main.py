import asyncio
import html as _html
import json
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from lxml import html as lxml_html
from playwright.sync_api import sync_playwright

load_dotenv()

from .agent import app_graph, parse_cv_structure
from .parser import extract_text


def _he(text) -> str:
    return _html.escape(str(text)) if text else ""

# ---------- Playwright (sync in thread — avoids Windows event loop issues) ----------

_pw = None
_browser = None
_browser_lock = threading.Lock()


def _init_browser():
    global _pw, _browser
    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch(headless=True)


def _close_browser():
    global _pw, _browser
    if _browser:
        _browser.close()
    if _pw:
        _pw.stop()


@asynccontextmanager
async def lifespan(app):
    try:
        await asyncio.to_thread(_init_browser)
    except Exception as e:
        print(f"WARNING: Browser init failed ({e}). PDF export and URL scraping will be unavailable.")
    yield
    await asyncio.to_thread(_close_browser)


# ---------- Job URL fetching ----------

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def _fetch_with_httpx(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=15, headers=_FETCH_HEADERS
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception:
        return None

    tree = lxml_html.fromstring(resp.content)
    for el in tree.xpath("//script|//style|//nav|//footer|//header|//aside"):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    text = re.sub(r"\s+", " ", tree.text_content()).strip()
    return text if len(text) >= 200 else None


def _playwright_scrape_sync(url: str) -> str:
    if _browser is None:
        raise RuntimeError("Browser not initialised.")
    with _browser_lock:
        page = _browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=25000)
            text = page.evaluate("() => document.body.innerText")
        finally:
            page.close()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 100:
        raise ValueError(
            "Page returned too little text — it may require a login. "
            "Please paste the job description directly."
        )
    return text[:8000]


async def fetch_job_from_url(url: str) -> str:
    text = await _fetch_with_httpx(url)
    if text:
        return text[:8000]

    if _browser is None:
        raise HTTPException(
            status_code=503,
            detail="Browser not available. Please paste the job description directly.",
        )
    try:
        return await asyncio.to_thread(_playwright_scrape_sync, url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not scrape job page: {e}")


# ---------- Export HTML builder ----------

def _build_export_html(payload: dict) -> str:
    company      = payload.get("company_name", "")
    role         = payload.get("role_title", "")
    score        = payload.get("match_score", 0)
    strong       = payload.get("strong_matches", [])
    gaps         = payload.get("skill_gaps", [])
    ats_kw       = payload.get("ats_keywords", [])
    gap_sugg     = payload.get("gap_suggestions", [])
    bullets      = payload.get("rewritten_bullets", [])
    cover        = payload.get("cover_letter", "")
    company_info = payload.get("company_info", "")

    def chips(items, color):
        return "".join(f'<span class="chip chip-{color}">{item}</span>' for item in items)

    def ats_rows():
        color_map = {"strong": "green", "weak": "amber", "absent": "red"}
        rows = []
        for k in ats_kw:
            c = color_map.get(k.get("status", "absent"), "red")
            rows.append(
                f"<tr>"
                f'<td class="ats-kw">{k.get("keyword","")}</td>'
                f'<td><span class="badge badge-{c}">{k.get("status","").capitalize()}</span></td>'
                f'<td><span class="badge badge-weight">{k.get("job_weight","")}</span></td>'
                f'<td class="ats-ctx">{k.get("context","")}</td>'
                f"</tr>"
            )
        return "".join(rows)

    def bullet_rows():
        rows = []
        for b in bullets:
            kws = "".join(
                f'<span class="chip chip-indigo">+{kw}</span>'
                for kw in (b.get("keywords_added") or [])
            )
            kws_html    = f'<div class="kws">{kws}</div>' if kws else ""
            reason_html = (
                f'<div class="bullet-reason">{b.get("reason","")}</div>'
                if b.get("reason") else ""
            )
            rows.append(
                f'<div class="bullet-block">'
                f'<div class="bullet-before"><span class="label">Before</span>{b.get("original","")}</div>'
                f'<div class="bullet-after"><span class="label">After</span>{b.get("rewritten","")}{kws_html}</div>'
                f"{reason_html}"
                f"</div>"
            )
        return "".join(rows)

    score_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 50 else "#f43f5e"

    gap_items = "".join(
        f'<div class="gap-item">'
        f'<span class="gap-skill">{s.get("skill","")}</span>'
        f'<p class="gap-text">{s.get("suggestion","")}</p>'
        f"</div>"
        for s in gap_sugg
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Job Analysis — {company} · {role}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;font-size:13px;color:#1e293b;background:#fff;padding:32px 40px;max-width:900px;margin:0 auto}}
  h1{{font-size:20px;font-weight:700;color:#1e293b}}
  h2{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin:24px 0 10px}}
  .meta{{color:#64748b;font-size:12px;margin-top:4px}}
  .score-row{{display:flex;align-items:center;gap:16px;margin:12px 0}}
  .score-circle{{width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:{score_color};border:5px solid {score_color};flex-shrink:0}}
  .chip{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;margin:2px 3px 2px 0}}
  .chip-green{{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}}
  .chip-red{{background:#fff1f2;color:#e11d48;border:1px solid #fecdd3}}
  .chip-indigo{{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe}}
  table{{width:100%;border-collapse:collapse;margin-top:6px}}
  th{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;text-align:left;padding:6px 8px;border-bottom:1px solid #e2e8f0}}
  td{{padding:8px;border-bottom:1px solid #f1f5f9;vertical-align:top}}
  .ats-kw{{font-weight:600;font-size:12px}}
  .ats-ctx{{color:#64748b;font-size:11px}}
  .badge{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:10px;font-weight:700}}
  .badge-green{{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}}
  .badge-amber{{background:#fffbeb;color:#b45309;border:1px solid #fde68a}}
  .badge-red{{background:#fff1f2;color:#e11d48;border:1px solid #fecdd3}}
  .badge-weight{{background:#f1f5f9;color:#475569;border:1px solid #e2e8f0}}
  .bullet-block{{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;margin-bottom:10px}}
  .bullet-before,.bullet-after{{padding:10px 14px}}
  .bullet-before{{background:#f8fafc;color:#64748b}}
  .bullet-after{{background:#eef2ff;color:#1e1b4b;font-weight:500}}
  .label{{display:block;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;opacity:.6}}
  .kws{{margin-top:6px}}
  .bullet-reason{{padding:6px 14px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;font-style:italic}}
  .cover{{white-space:pre-wrap;line-height:1.8;color:#334155;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px}}
  .section{{margin-bottom:28px}}
  .gap-item{{display:flex;gap:12px;padding:10px 14px;background:#fffbeb;border:1px solid #fde68a;border-radius:10px;margin-bottom:8px}}
  .gap-skill{{font-size:10px;font-weight:700;background:#fef3c7;color:#b45309;border:1px solid #fde68a;padding:2px 8px;border-radius:999px;white-space:nowrap;height:fit-content;margin-top:1px}}
  .gap-text{{font-size:12px;color:#64748b;line-height:1.5}}
  @media print{{body{{padding:20px 24px}}.no-print{{display:none}}}}
</style>
</head>
<body>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px">
    <div>
      <h1>{company or "Job Analysis"}</h1>
      <p class="meta">{role}</p>
      {f'<p class="meta" style="margin-top:6px;max-width:600px">{company_info}</p>' if company_info else ""}
    </div>
    <button class="no-print" onclick="window.print()" style="padding:8px 18px;background:#4f46e5;color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer">Print / Save PDF</button>
  </div>

  <div class="section">
    <h2>Fit Score</h2>
    <div class="score-row">
      <div class="score-circle">{score}</div>
      <div>
        <div style="margin-bottom:6px">{chips(strong, "green")}</div>
        <div>{chips(gaps, "red")}</div>
      </div>
    </div>
  </div>

  {"<div class='section'><h2>ATS Keyword Analysis</h2><table><tr><th>Keyword</th><th>Status</th><th>Weight</th><th>CV Context</th></tr>" + ats_rows() + "</table></div>" if ats_kw else ""}

  {"<div class='section'><h2>Gap Suggestions</h2>" + gap_items + "</div>" if gap_sugg else ""}

  {"<div class='section'><h2>CV Rewrites</h2>" + bullet_rows() + "</div>" if bullets else ""}

  {"<div class='section'><h2>Cover Letter</h2><div class='cover'>" + cover + "</div></div>" if cover else ""}
</body>
</html>"""


# ---------- App ----------

app = FastAPI(title="Job Application Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/")
async def root():
    return FileResponse(_frontend)


@app.post("/analyse")
async def analyse(
    cv_file: UploadFile = File(...),
    job_description: str = Form(""),
    job_url: str = Form(""),
):
    file_bytes = await cv_file.read()
    filename = cv_file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        cv_text = extract_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CV: {e}")

    if job_url.strip():
        job_description = await fetch_job_from_url(job_url.strip())
    elif not job_description.strip():
        raise HTTPException(status_code=400, detail="Provide either a job URL or a job description.")

    initial_state = {
        "cv_text": cv_text,
        "job_description": job_description,
        "extracted_info": {},
        "company_info": "",
        "fit_analysis": {},
        "gap_suggestions": [],
        "rewritten_bullets": [],
        "cover_letter": "",
    }

    async def stream():
        yield json.dumps({"type": "cv_text", "data": cv_text}) + "\n"
        yield json.dumps({"type": "progress", "step": "Extracting information..."}) + "\n"

        # Track which parallel nodes have completed so we can emit the next
        # progress event only after the last sibling in each parallel group finishes.
        stage3_done: set[str] = set()  # suggest_gaps, rewrite_bullets

        try:
            async for event in app_graph.astream(initial_state):
                # Parallel nodes may arrive in the same event dict or separately;
                # iterate over every key to handle both cases.
                for node_name, node_output in event.items():

                    if node_name == "extract_info":
                        yield json.dumps({
                            "type": "extracted_info",
                            "data": node_output.get("extracted_info", {}),
                        }) + "\n"
                        # research_company and analyse_fit now start in parallel
                        yield json.dumps({"type": "progress", "step": "Researching & analysing..."}) + "\n"

                    elif node_name == "research_company":
                        yield json.dumps({
                            "type": "company_info",
                            "data": node_output.get("company_info", ""),
                        }) + "\n"

                    elif node_name == "analyse_fit":
                        yield json.dumps({
                            "type": "fit_analysis",
                            "data": node_output.get("fit_analysis", {}),
                        }) + "\n"
                        # suggest_gaps and rewrite_bullets now start in parallel
                        yield json.dumps({"type": "progress", "step": "Generating suggestions & rewrites..."}) + "\n"

                    elif node_name == "suggest_gaps":
                        yield json.dumps({
                            "type": "gap_suggestions",
                            "data": node_output.get("gap_suggestions", []),
                        }) + "\n"
                        stage3_done.add("suggest_gaps")
                        if "rewrite_bullets" in stage3_done:
                            yield json.dumps({"type": "progress", "step": "Writing cover letter..."}) + "\n"

                    elif node_name == "rewrite_bullets":
                        yield json.dumps({
                            "type": "rewritten_bullets",
                            "data": node_output.get("rewritten_bullets", []),
                        }) + "\n"
                        stage3_done.add("rewrite_bullets")
                        if "suggest_gaps" in stage3_done:
                            yield json.dumps({"type": "progress", "step": "Writing cover letter..."}) + "\n"

                    elif node_name == "generate_cover_letter":
                        yield json.dumps({
                            "type": "cover_letter",
                            "data": node_output.get("cover_letter", ""),
                        }) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _build_resume_editor_html(s: dict, api_base: str) -> str:
    """Build a resume editor page using the Resume-Matcher Swiss Single template."""

    def contact_line(c: dict) -> str:
        parts = [c.get("email",""), c.get("phone",""), c.get("location",""), c.get("website","")]
        return ", ".join(p for p in parts if p)

    def exp_blocks() -> str:
        blocks = []
        for e in s.get("experience", []):
            bullets_html = "".join(
                f'<li contenteditable="true" class="cv-bullet">{_he(b)}</li>'
                for b in e.get("bullets", [])
            )
            kw = _he(e.get("keywords", ""))
            kw_html = f'<p contenteditable="true" class="cv-keywords">{kw}</p>' if kw else ""
            meta_parts = [x for x in [e.get("location",""), e.get("dates","")] if x]
            meta_str = _he(" · ".join(meta_parts))
            blocks.append(f"""
            <div class="cv-exp">
              <div class="cv-exp-top">
                <span contenteditable="true" class="cv-company">{_he(e.get("company",""))}</span>
                <span contenteditable="true" class="cv-meta">{meta_str}</span>
              </div>
              <div contenteditable="true" class="cv-jobtitle">{_he(e.get("title",""))}</div>
              <ul class="cv-bullets">{bullets_html}</ul>
              {kw_html}
            </div>""")
        return "".join(blocks)

    def edu_blocks() -> str:
        blocks = []
        for e in s.get("education", []):
            meta_parts = [x for x in [e.get("location",""), e.get("dates","")] if x]
            meta_str = _he(" · ".join(meta_parts))
            blocks.append(f"""
            <div class="cv-edu">
              <div class="cv-exp-top">
                <span contenteditable="true" class="cv-company">{_he(e.get("institution",""))}</span>
                <span contenteditable="true" class="cv-meta">{meta_str}</span>
              </div>
              <div contenteditable="true" class="cv-jobtitle">{_he(e.get("degree",""))}</div>
            </div>""")
        return "".join(blocks)

    def project_blocks() -> str:
        blocks = []
        for p in s.get("projects", []):
            blocks.append(f"""
            <div class="cv-project">
              <div class="cv-exp-top">
                <span contenteditable="true" class="cv-company">{_he(p.get("name",""))}</span>
                <span contenteditable="true" class="cv-meta">{_he(p.get("date",""))}</span>
              </div>
              <div contenteditable="true" class="cv-proj-desc">{_he(p.get("description",""))}</div>
            </div>""")
        return "".join(blocks)

    summary_section = f"""
      <div class="cv-section">
        <div class="cv-section-title">Summary</div>
        <div contenteditable="true" class="cv-body-text">{_he(s.get("summary",""))}</div>
      </div>""" if s.get("summary") else ""

    skills_section = f"""
      <div class="cv-section">
        <div class="cv-section-title">Technical Skills</div>
        <div contenteditable="true" class="cv-body-text">{_he(s.get("skills",""))}</div>
      </div>""" if s.get("skills") else ""

    projects_section = f"""
      <div class="cv-section">
        <div class="cv-section-title">Projects</div>
        {project_blocks()}
      </div>""" if s.get("projects") else ""

    languages_section = f"""
      <div class="cv-section">
        <div class="cv-section-title">Languages</div>
        <div contenteditable="true" class="cv-body-text">{_he(s.get("languages",""))}</div>
      </div>""" if s.get("languages") else ""

    # CSS for PDF/print export — plain string so braces don't need escaping
    # @page margin:0 suppresses the browser's URL/date header and footer
    print_css = (
        "@page{size:A4;margin:0}\n"
        "*{box-sizing:border-box;margin:0;padding:0}\n"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;color:#000}\n"
        ".cv-page{padding:12mm 14mm;-webkit-box-decoration-break:clone;box-decoration-break:clone}\n"
        ".cv-header{text-align:center;border-bottom:1px solid #9ca3af;padding-bottom:.5rem;margin-bottom:.65rem}\n"
        ".cv-name{display:block;font-family:ui-serif,Georgia,Cambria,'Times New Roman',Times,serif;"
        "font-size:26px;font-weight:700;text-transform:uppercase;letter-spacing:-.01em;line-height:1.1}\n"
        ".cv-title-line{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:.1em;margin-top:4px}\n"
        ".cv-contact-line{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:11px;color:#4b5563;margin-top:5px}\n"
        ".cv-section{margin-top:.65rem;break-inside:avoid}\n"
        ".cv-section-title{font-family:ui-serif,Georgia,Cambria,'Times New Roman',Times,serif;"
        "font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid #9ca3af;padding-bottom:2px;margin-bottom:.35rem}\n"
        ".cv-body-text{font-size:13px;line-height:1.5;color:#374151}\n"
        ".cv-exp{margin-bottom:.4rem;break-inside:avoid}.cv-edu{margin-bottom:.35rem;break-inside:avoid}.cv-project{margin-bottom:.35rem;break-inside:avoid}\n"
        ".cv-exp-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}\n"
        ".cv-company{font-size:13px;font-weight:700}\n"
        ".cv-meta{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:11px;color:#4b5563;white-space:nowrap;flex-shrink:0}\n"
        ".cv-jobtitle{font-size:13px;font-weight:600;font-style:italic;margin-top:.1rem;color:#374151}\n"
        ".cv-bullets{list-style:disc;padding-left:1.1rem;margin-top:.2rem}\n"
        ".cv-bullet{font-size:13px;line-height:1.5;color:#374151;margin-bottom:.1rem}\n"
        ".cv-proj-desc{font-size:13px;line-height:1.5;color:#374151;margin-top:.1rem}\n"
        ".cv-keywords{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:11px;color:#4b5563;margin-top:.2rem}\n"
        ".cv-skills-row{display:flex;gap:.5rem;align-items:flex-start}\n"
        ".cv-skill-label{font-weight:700;width:100px;flex-shrink:0;font-size:13px}\n"
        ".cv-skill-content{font-size:13px;line-height:1.5;color:#374151;flex:1}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Resume — {_he(s.get("name",""))}</title>
<style>
  :root {{
    --font-serif: ui-serif, Georgia, Cambria, 'Times New Roman', Times, serif;
    --font-sans: ui-sans-serif, system-ui, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --text-primary: #000000;
    --text-secondary: #374151;
    --text-tertiary: #4b5563;
    --border-primary: #9ca3af;
    --font-base: 13px;
    --font-name: 26px;
    --font-section: 14px;
    --font-meta: 11px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: var(--font-sans); font-size: var(--font-base); background: #f3f4f6; color: var(--text-primary); }}

  .toolbar {{ position: fixed; top: 0; left: 0; right: 0; height: 52px; background: #1f2937; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; z-index: 100; }}
  .toolbar-left {{ display: flex; align-items: center; gap: 12px; }}
  .toolbar-title {{ color: #fff; font-size: 13px; font-weight: 600; }}
  .toolbar-hint {{ color: rgba(255,255,255,.5); font-size: 11px; }}
  .toolbar-right {{ display: flex; gap: 8px; }}
  .tb-btn {{ padding: 7px 16px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; border: none; }}
  .tb-btn-primary {{ background: #fff; color: #1f2937; }}
  .tb-btn-primary:hover {{ background: #e5e7eb; }}
  .tb-btn-secondary {{ background: transparent; color: #fff; border: 1px solid rgba(255,255,255,.3); }}
  .tb-btn-secondary:hover {{ background: rgba(255,255,255,.1); }}
  .tb-btn:disabled {{ opacity: .5; cursor: not-allowed; }}

  .page-wrap {{ padding: 68px 24px 48px; display: flex; justify-content: center; }}
  .cv-page {{ background: #fff; width: 794px; min-height: 1123px; padding: 10mm; box-shadow: 0 4px 24px rgba(0,0,0,.08); position: relative; }}
  .cv-page::after {{ content: ''; position: absolute; left: 0; right: 0; top: 1123px; border-top: 2px dashed #cbd5e1; pointer-events: none; }}
  .cv-page::before {{ content: 'Page 2'; position: absolute; right: 10mm; top: 1130px; font-size: 10px; color: #94a3b8; pointer-events: none; }}

  [contenteditable] {{ border-radius: 2px; outline: none; transition: background .1s; }}
  [contenteditable]:hover {{ background: rgba(156,163,175,.1); }}
  [contenteditable]:focus {{ background: rgba(156,163,175,.18); box-shadow: 0 0 0 1.5px var(--border-primary); }}

  .cv-header {{ text-align: center; border-bottom: 1px solid var(--border-primary); padding-bottom: .5rem; margin-bottom: .65rem; }}
  .cv-name {{ display: block; font-family: var(--font-serif); font-size: var(--font-name); font-weight: 700; text-transform: uppercase; letter-spacing: -.01em; line-height: 1.1; }}
  .cv-title-line {{ display: block; font-family: var(--font-mono); font-size: var(--font-meta); color: var(--text-tertiary); text-transform: uppercase; letter-spacing: .1em; margin-top: 4px; }}
  .cv-contact-line {{ display: block; font-family: var(--font-mono); font-size: var(--font-meta); color: var(--text-tertiary); margin-top: 6px; }}

  .cv-section {{ margin-top: .65rem; }}
  .cv-section-title {{ font-family: var(--font-serif); font-size: var(--font-section); font-weight: 700; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border-primary); padding-bottom: 2px; margin-bottom: .35rem; }}

  .cv-body-text {{ font-size: var(--font-base); line-height: 1.6; color: var(--text-secondary); }}

  .cv-exp {{ margin-bottom: .4rem; }}
  .cv-edu {{ margin-bottom: .35rem; }}
  .cv-project {{ margin-bottom: .35rem; }}
  .cv-exp-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }}
  .cv-company {{ font-size: var(--font-base); font-weight: 700; }}
  .cv-meta {{ font-family: var(--font-mono); font-size: var(--font-meta); color: var(--text-tertiary); white-space: nowrap; flex-shrink: 0; }}
  .cv-jobtitle {{ font-size: var(--font-base); font-weight: 700; font-style: italic; margin-top: .125rem; color: var(--text-secondary); }}
  .cv-bullets {{ list-style: disc; padding-left: 1.25rem; margin-top: .25rem; }}
  .cv-bullet {{ font-size: var(--font-base); line-height: 1.5; color: var(--text-secondary); margin-bottom: .1rem; padding: 1px 2px; }}
  .cv-proj-desc {{ font-size: var(--font-base); line-height: 1.55; color: var(--text-secondary); margin-top: .125rem; }}
  .cv-keywords {{ font-family: var(--font-mono); font-size: var(--font-meta); color: var(--text-tertiary); margin-top: .25rem; }}

  .cv-skills-row {{ display: flex; gap: .5rem; align-items: flex-start; }}
  .cv-skill-label {{ font-weight: 700; width: 128px; flex-shrink: 0; font-size: var(--font-base); }}
  .cv-skill-content {{ font-size: var(--font-base); line-height: 1.6; color: var(--text-secondary); flex: 1; }}

  @media print {{
    .toolbar {{ display: none; }}
    body {{ background: #fff; }}
    .page-wrap {{ padding: 0; }}
    .cv-page {{ box-shadow: none; width: 100%; min-height: auto; padding: 12mm 14mm; }}
    [contenteditable]:hover, [contenteditable]:focus {{ background: transparent; box-shadow: none; }}
  }}
</style>
</head>
<body>

<div class="toolbar">
  <div class="toolbar-left">
    <span class="toolbar-title">Resume Editor</span>
    <span class="toolbar-hint">Click any text to edit &nbsp;·&nbsp; In print dialog, uncheck "Headers and footers"</span>
  </div>
  <div class="toolbar-right">
    <button class="tb-btn tb-btn-secondary" onclick="window.print()">Print</button>
    <button class="tb-btn tb-btn-primary" id="dlBtn" onclick="downloadPdf()">Download PDF</button>
  </div>
</div>

<div class="page-wrap">
  <div class="cv-page" id="cv-page">

    <div class="cv-header">
      <span contenteditable="true" class="cv-name">{_he(s.get("name",""))}</span>
      <span contenteditable="true" class="cv-title-line">{_he(s.get("title",""))}</span>
      <span contenteditable="true" class="cv-contact-line">{_he(contact_line(s.get("contact",{})))}</span>
    </div>

    {summary_section}

    {"<div class='cv-section'><div class='cv-section-title'>Experience</div>" + exp_blocks() + "</div>" if s.get("experience") else ""}

    {"<div class='cv-section'><div class='cv-section-title'>Education</div>" + edu_blocks() + "</div>" if s.get("education") else ""}

    {skills_section}

    {projects_section}

    {languages_section}

  </div>
</div>

<script>
const API = '{api_base}';

function downloadPdf() {{
  const html = buildPrintHtml();
  const blob = new Blob([html], {{type: 'text/html'}});
  const url = URL.createObjectURL(blob);
  const w = window.open(url, '_blank');
  if (!w) {{ alert('Allow pop-ups for this site to download the PDF.'); return; }}
  w.addEventListener('load', () => {{
    setTimeout(() => {{
      w.print();
      URL.revokeObjectURL(url);
    }}, 300);
  }});
}}

function buildPrintHtml() {{
  const page = document.getElementById('cv-page').cloneNode(true);
  page.querySelectorAll('[contenteditable]').forEach(el => el.removeAttribute('contenteditable'));
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<style>
{print_css}
</style></head><body>` + page.outerHTML + `</body></html>`;
}}
</script>
</body>
</html>"""


@app.post("/build-resume")
async def build_resume(payload: dict, request: Request):
    cv_text          = payload.get("cv_text", "")
    rewritten_bullets = payload.get("rewritten_bullets", [])

    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="No CV text provided.")

    structure = await asyncio.to_thread(parse_cv_structure, cv_text, rewritten_bullets)
    api_base  = str(request.base_url).rstrip("/")
    return HTMLResponse(content=_build_resume_editor_html(structure, api_base))


def _render_pdf(html: str, margin: dict) -> bytes:
    with _browser_lock:
        page = _browser.new_page()
        try:
            page.set_content(html, wait_until="domcontentloaded")
            return page.pdf(format="A4", print_background=True, margin=margin)
        finally:
            page.close()


@app.post("/resume-pdf")
async def resume_pdf(payload: dict):
    html = payload.get("html", "")
    if not html.strip():
        raise HTTPException(status_code=400, detail="No HTML provided.")
    if _browser is None:
        raise HTTPException(status_code=503, detail="Browser not available.")

    pdf_bytes = await asyncio.to_thread(
        _render_pdf, html, {"top": "0", "bottom": "0", "left": "0", "right": "0"}
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="resume.pdf"'},
    )


@app.post("/export-html")
async def export_html(payload: dict):
    html = _build_export_html(payload)
    return HTMLResponse(content=html)
