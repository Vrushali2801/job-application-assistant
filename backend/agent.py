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
    temperature=0.3,
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


def _parse_json(text: str):
    text = text.strip()
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith(("{", "[")):
                text = candidate
                break
    # Find the first JSON object or array
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start != -1:
            # Walk backwards from end to find matching close
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
{state['cv_text'][:3000]}

Job Description:
{state['job_description'][:2000]}

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{
  "company_name": "company name from job description",
  "role_title": "job title being applied for",
  "required_skills": ["skill1", "skill2", "skill3"],
  "cv_bullets": ["bullet point 1", "bullet point 2"]
}}

For cv_bullets extract all experience and achievement statements from the CV (max 15)."""

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

    prompt = f"""Analyse how well this candidate fits the job.

Candidate experience (CV bullets):
{chr(10).join(f"- {b}" for b in cv_bullets[:15])}

Job required skills:
{chr(10).join(f"- {s}" for s in required_skills)}

Job description excerpt:
{state['job_description'][:1500]}

Return ONLY a JSON object (no markdown, no explanation):
{{
  "match_score": 72,
  "strong_matches": ["skill or experience that aligns well"],
  "skill_gaps": ["skill the candidate lacks or needs to strengthen"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"]
}}

match_score is 0-100. missing_keywords are important terms from the job description absent from the CV."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        fit = _parse_json(response.content)
        if not isinstance(fit, dict):
            raise ValueError("Not a dict")
        fit.setdefault("match_score", 50)
        fit.setdefault("strong_matches", [])
        fit.setdefault("skill_gaps", [])
        fit.setdefault("missing_keywords", [])
    except Exception:
        fit = {
            "match_score": 50,
            "strong_matches": [],
            "skill_gaps": [],
            "missing_keywords": [],
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

    bullets_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(cv_bullets[:12]))
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


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_info", extract_info)
    graph.add_node("research_company", research_company)
    graph.add_node("analyse_fit", analyse_fit)
    graph.add_node("suggest_gaps", suggest_gaps)
    graph.add_node("rewrite_bullets", rewrite_bullets)

    graph.add_edge(START, "extract_info")
    graph.add_edge("extract_info", "research_company")
    graph.add_edge("research_company", "analyse_fit")
    graph.add_edge("analyse_fit", "suggest_gaps")
    graph.add_edge("suggest_gaps", "rewrite_bullets")
    graph.add_edge("rewrite_bullets", END)

    return graph.compile()


app_graph = build_graph()
