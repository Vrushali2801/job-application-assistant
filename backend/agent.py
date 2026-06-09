import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from .tools import web_search

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


class AgentState(TypedDict):
    cv_text: str
    job_description: str
    extracted_info: dict
    company_info: str
    fit_analysis: dict
    gap_suggestions: list
    rewritten_bullets: list
    cover_letter: str


def _parse_json(text: str):
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith(("{", "[")):
                text = candidate
                break
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start != -1:
            end = text.rfind(end_char)
            if end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
    return json.loads(text)


def extract_info(state: AgentState) -> dict:
    prompt = f"""Extract key information from this CV and job description.

CV:
{state['cv_text'][:5000]}

Job Description:
{state['job_description'][:2000]}

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{
  "company_name": "company name from job description",
  "role_title": "job title being applied for",
  "required_skills": ["skill1", "skill2", "skill3"],
  "cv_bullets": ["bullet point 1", "bullet point 2"]
}}

For cv_bullets extract ALL experience and achievement statements from the CV (no cap — include every bullet, role description, and achievement)."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        extracted = _parse_json(response.content)
        if not isinstance(extracted, dict):
            raise ValueError("Not a dict")
    except Exception:
        extracted = {
            "company_name": "Unknown",
            "role_title": "Unknown",
            "required_skills": [],
            "cv_bullets": [],
        }
    return {"extracted_info": extracted}


def research_company(state: AgentState) -> dict:
    company = state["extracted_info"].get("company_name", "").strip()
    if not company or company.lower() in ("unknown", ""):
        return {"company_info": ""}

    results = web_search(f"{company} company overview mission culture")
    if not results:
        return {"company_info": ""}

    prompt = f"""Based on these search results about {company}, write a concise 2-3 sentence summary covering what the company does and its culture/values.

Search results:
{results[:2000]}

Write only the summary, no preamble or labels."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"company_info": response.content.strip()}
    except Exception:
        return {"company_info": results[:300]}


def analyse_fit(state: AgentState) -> dict:
    cv_bullets = state["extracted_info"].get("cv_bullets", [])
    required_skills = state["extracted_info"].get("required_skills", [])

    prompt = f"""Analyse how well this candidate fits the job. Be precise about context and seniority — not just whether a skill appears, but how it appears.

Full CV text (use this as the ground truth for keyword presence):
{state['cv_text'][:5000]}

Candidate experience bullets (for assessing depth of experience):
{chr(10).join(f"- {b}" for b in cv_bullets[:20])}

Job required skills:
{chr(10).join(f"- {s}" for s in required_skills)}

Job description excerpt:
{state['job_description'][:1500]}

Return ONLY a JSON object (no markdown, no explanation):
{{
  "match_score": 72,
  "strong_matches": ["skill or experience that aligns well with evidence"],
  "skill_gaps": ["skill the candidate lacks or needs to strengthen"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "ats_keywords": [
    {{
      "keyword": "Python",
      "status": "strong",
      "context": "Used professionally for 3+ years per CV",
      "job_weight": "required"
    }},
    {{
      "keyword": "Kubernetes",
      "status": "weak",
      "context": "Mentioned once without depth",
      "job_weight": "preferred"
    }},
    {{
      "keyword": "AWS",
      "status": "absent",
      "context": "Not mentioned in CV",
      "job_weight": "required"
    }}
  ]
}}

Rules:
- match_score is 0-100
- missing_keywords: important terms from the job description that are fully absent from the CV
- ats_keywords: list the top 8-10 most important keywords from the job description, each with:
  - status: "strong" (backed by concrete experience in the bullets), "weak" (mentioned somewhere in the CV but without demonstrated depth), or "absent" (does not appear anywhere in the full CV text)
  - context: one short phrase explaining the evidence (or lack of it)
  - job_weight: "required" or "preferred" based on the job description language
- IMPORTANT: search the FULL CV TEXT to determine presence. A keyword is "absent" only if it genuinely does not appear anywhere in the CV — not in bullets, not in a skills list, not in project descriptions, nowhere."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        fit = _parse_json(response.content)
        if not isinstance(fit, dict):
            raise ValueError("Not a dict")
        fit.setdefault("match_score", 50)
        fit.setdefault("strong_matches", [])
        fit.setdefault("skill_gaps", [])
        fit.setdefault("missing_keywords", [])
        fit.setdefault("ats_keywords", [])
    except Exception:
        fit = {
            "match_score": 50,
            "strong_matches": [],
            "skill_gaps": [],
            "missing_keywords": [],
            "ats_keywords": [],
        }
    return {"fit_analysis": fit}


def suggest_gaps(state: AgentState) -> dict:
    skill_gaps = state["fit_analysis"].get("skill_gaps", [])
    if not skill_gaps:
        return {"gap_suggestions": []}

    gaps_text = "\n".join(f"- {g}" for g in skill_gaps)

    prompt = f"""A candidate is applying for this role but has gaps in the following skills:

{gaps_text}

Job description context:
{state['job_description'][:1000]}

Candidate background (from their CV):
{state['cv_text'][:1500]}

For each skill gap, suggest one concrete project or hands-on thing the candidate could add to their portfolio or experience to demonstrate that skill. The suggestion must:
- Be a specific, buildable project or a concrete skill they could practise and demonstrate
- Be relevant to the candidate's existing background — don't suggest something completely unrelated to what they already do
- NOT mention courses, certifications, or studying — only things they can build or do
- Be concise (1-2 sentences max)

If a skill gap is completely unrelated to the candidate's field and no realistic project suggestion exists, skip it.

Return ONLY a JSON array (no markdown, no explanation):
[
  {{
    "skill": "the skill gap",
    "suggestion": "specific project or hands-on thing to do"
  }}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        suggestions = _parse_json(response.content)
        if not isinstance(suggestions, list):
            raise ValueError("Not a list")
    except Exception:
        suggestions = []
    return {"gap_suggestions": suggestions}


def rewrite_bullets(state: AgentState) -> dict:
    cv_bullets = state["extracted_info"].get("cv_bullets", [])
    missing_keywords = state["fit_analysis"].get("missing_keywords", [])

    if not cv_bullets:
        return {"rewritten_bullets": []}

    bullets_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(cv_bullets[:20]))
    keywords_text = ", ".join(missing_keywords[:10]) if missing_keywords else "none identified"

    prompt = f"""You are reviewing CV bullet points against a job description. Your job is to identify which bullets genuinely benefit from rewriting and improve only those.

Original bullets:
{bullets_text}

Job description:
{state['job_description'][:1500]}

Keywords from the job that are missing from the CV:
{keywords_text}

RULES:
- Only include a bullet in your response if it genuinely benefits from a change. If a bullet is already strong, well-phrased, and relevant — skip it entirely. Do not return it.
- A bullet benefits from rewriting if: the language is weak or vague, the framing could better reflect the role's priorities, or a missing keyword accurately applies to what it describes.
- Do NOT invent skills, technologies, domains, or achievements not present in the original.
- Keep all existing numbers and metrics exactly as they are.
- Only add a keyword if it genuinely and accurately describes what the bullet already talks about. If it doesn't fit, do not add it — leave keywords_added empty.
- Never force a change just to produce output. Fewer high-quality rewrites is better than many unnecessary ones.

Return ONLY a JSON array of bullets that were actually improved (may be empty if none need changing). No markdown, no explanation:
[
  {{
    "original": "exact original bullet text",
    "rewritten": "improved bullet text",
    "keywords_added": [],
    "reason": "specific reason this bullet needed improving"
  }}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        rewrites = _parse_json(response.content)
        if not isinstance(rewrites, list):
            raise ValueError("Not a list")
    except Exception:
        rewrites = [
            {"original": b, "rewritten": b, "keywords_added": [], "reason": ""}
            for b in cv_bullets[:12]
        ]
    return {"rewritten_bullets": rewrites}


def generate_cover_letter(state: AgentState) -> dict:
    company = state["extracted_info"].get("company_name", "the company")
    role = state["extracted_info"].get("role_title", "the role")
    strong_matches = state["fit_analysis"].get("strong_matches", [])
    company_info = state.get("company_info", "")

    prompt = f"""Write a professional cover letter body for this job application.

Candidate CV:
{state['cv_text'][:2500]}

Job Description:
{state['job_description'][:1500]}

Company: {company}
Role: {role}
{f"Company background: {company_info}" if company_info else ""}
Candidate's strongest relevant skills: {", ".join(strong_matches[:5]) if strong_matches else "see CV"}

RULES:
- 3 paragraphs: (1) opening — why this specific role and company, (2) middle — 2-3 concrete achievements from their CV that are directly relevant to this role, (3) closing — brief call to action
- Do NOT use stale openers like "I am writing to express my interest" or "I believe I would be a great fit"
- Reference specific details from the job description or company
- Do NOT invent skills or achievements not present in the CV
- Keep under 280 words
- Do not include date, address headers, salutation, or sign-off — body paragraphs only
- First person, professional but direct

Return only the cover letter body text, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"cover_letter": response.content.strip()}


def parse_cv_structure(cv_text: str, rewritten_bullets: list) -> dict:
    rewrites: dict[str, str] = {}
    for b in rewritten_bullets:
        orig = (b.get("original") or "").strip()
        rew  = (b.get("rewritten") or "").strip()
        if orig and rew:
            rewrites[orig] = rew

    prompt = f"""Parse this CV into structured JSON. Work with any CV format — not all CVs have every section.

CV:
{cv_text[:8000]}

Return ONLY valid JSON (no markdown):
{{
  "name": "Full Name",
  "title": "Professional Title",
  "contact": {{
    "email": "",
    "phone": "",
    "location": "",
    "website": ""
  }},
  "summary": "summary paragraph or empty string",
  "experience": [
    {{
      "company": "",
      "location": "",
      "title": "",
      "dates": "",
      "bullets": ["bullet 1", "bullet 2"],
      "keywords": "Keywords line if present, else empty string"
    }}
  ],
  "education": [
    {{
      "institution": "",
      "location": "",
      "degree": "",
      "dates": ""
    }}
  ],
  "skills": "full skills text exactly as written in CV, or empty string",
  "projects": [
    {{
      "name": "",
      "date": "",
      "description": ""
    }}
  ],
  "languages": "languages text or empty string"
}}

Rules: use "" for missing text, [] for missing arrays. Preserve exact wording."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        structure = _parse_json(response.content)
        if not isinstance(structure, dict):
            raise ValueError
    except Exception:
        structure = {
            "name": "", "title": "",
            "contact": {"email": "", "phone": "", "location": "", "website": ""},
            "summary": "", "experience": [], "education": [],
            "skills": "", "projects": [], "languages": "",
        }

    # Apply rewrites to matching bullets (exact then prefix match)
    for exp in structure.get("experience", []):
        new_bullets = []
        for bullet in exp.get("bullets", []):
            stripped = bullet.strip()
            if stripped in rewrites:
                new_bullets.append(rewrites[stripped])
                continue
            matched = None
            for orig, rew in rewrites.items():
                if len(orig) > 20 and (orig[:60] in bullet or bullet[:60] in orig):
                    matched = rew
                    break
            new_bullets.append(matched if matched else bullet)
        exp["bullets"] = new_bullets

    return structure


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_info", extract_info)
    graph.add_node("research_company", research_company)
    graph.add_node("analyse_fit", analyse_fit)
    graph.add_node("suggest_gaps", suggest_gaps)
    graph.add_node("rewrite_bullets", rewrite_bullets)
    graph.add_node("generate_cover_letter", generate_cover_letter)

    graph.add_edge(START, "extract_info")

    # Fan-out: research_company and analyse_fit run in parallel after extract_info
    graph.add_edge("extract_info", "research_company")
    graph.add_edge("extract_info", "analyse_fit")

    # Fan-out: suggest_gaps and rewrite_bullets run in parallel after analyse_fit
    graph.add_edge("analyse_fit", "suggest_gaps")
    graph.add_edge("analyse_fit", "rewrite_bullets")

    # Fan-in: generate_cover_letter waits for all three to complete
    graph.add_edge("research_company", "generate_cover_letter")
    graph.add_edge("suggest_gaps", "generate_cover_letter")
    graph.add_edge("rewrite_bullets", "generate_cover_letter")

    graph.add_edge("generate_cover_letter", END)

    return graph.compile()


app_graph = build_graph()
