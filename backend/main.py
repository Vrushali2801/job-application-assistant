import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

load_dotenv()

from .agent import app_graph
from .parser import extract_text

app = FastAPI(title="Job Application Assistant")

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
    job_description: str = Form(...),
):
    file_bytes = await cv_file.read()
    filename = cv_file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        cv_text = extract_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CV: {e}")

    initial_state = {
        "cv_text": cv_text,
        "job_description": job_description,
        "extracted_info": {},
        "company_info": "",
        "fit_analysis": {},
        "gap_suggestions": [],
        "rewritten_bullets": [],
    }

    async def stream():
        yield json.dumps({"type": "progress", "step": "Extracting information..."}) + "\n"
        try:
            async for event in app_graph.astream(initial_state):
                node_name = next(iter(event))
                node_output = event[node_name]

                if node_name == "extract_info":
                    yield json.dumps({
                        "type": "extracted_info",
                        "data": node_output.get("extracted_info", {}),
                    }) + "\n"
                    yield json.dumps({"type": "progress", "step": "Researching company..."}) + "\n"

                elif node_name == "research_company":
                    yield json.dumps({
                        "type": "company_info",
                        "data": node_output.get("company_info", ""),
                    }) + "\n"
                    yield json.dumps({"type": "progress", "step": "Analysing fit..."}) + "\n"

                elif node_name == "analyse_fit":
                    yield json.dumps({
                        "type": "fit_analysis",
                        "data": node_output.get("fit_analysis", {}),
                    }) + "\n"
                    yield json.dumps({"type": "progress", "step": "Generating suggestions..."}) + "\n"

                elif node_name == "suggest_gaps":
                    yield json.dumps({
                        "type": "gap_suggestions",
                        "data": node_output.get("gap_suggestions", []),
                    }) + "\n"
                    yield json.dumps({"type": "progress", "step": "Rewriting bullets..."}) + "\n"

                elif node_name == "rewrite_bullets":
                    yield json.dumps({
                        "type": "rewritten_bullets",
                        "data": node_output.get("rewritten_bullets", []),
                    }) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
