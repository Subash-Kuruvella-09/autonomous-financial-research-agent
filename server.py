"""
ARA-1 Financial Research Agent — Web Server

FastAPI server that wraps the existing agent backend:
  - POST /api/chat     — Chat endpoint (handles chitchat, fast queries, research, PDF)
  - GET  /api/download/{filename} — Download generated PDFs
  - GET  /              — Serves the chat frontend

Usage:
  python server.py
"""

import os
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="ARA-1 Financial Research Agent",
    description="Autonomous Financial Intelligence — Chat API",
    version="1.0.0",
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Request / Response Models ──────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    report: str
    ticker: str | None = None
    company_name: str | None = None
    intent: str | None = None
    pdf_url: str | None = None


# ─── Agent Wrapper (runs in thread to avoid blocking) ───────

def _run_agent_sync(user_query: str) -> dict:
    """
    Synchronous wrapper that calls the existing agent pipeline.
    
    Handles:
      1. Chitchat → casual response
      2. PDF request → full report + PDF file generation
      3. Financial query → agent response (FAST or RESEARCH path)
    """
    from agent.core import run_agent
    from tools.report_gen import generate_report, clean_output
    from tools.company_profile import company_profile

    wants_pdf = "pdf" in user_query.lower()

    if wants_pdf:
        # Full pipeline: run_agent → clean → generate PDF
        report_data = generate_report(user_query, user_query)
        report_text = report_data.get("report", "")
        ticker = report_data.get("ticker", "Unknown")
        company_name = report_data.get("company_name", ticker)
        intent = report_data.get("intent", "RESEARCH")

        pdf_url = None
        # Only generate PDF if the report is meaningful
        if (report_text
                and not report_text.startswith("I encountered a technical issue")
                and not report_text.startswith("I have gathered the data")
                and len(report_text) > 50):
            try:
                from utils.pdf_generator import generate_pdf
                pdf_path = generate_pdf(report_text, company_name)
                pdf_filename = os.path.basename(pdf_path)
                pdf_url = f"/api/download/{pdf_filename}"
            except Exception as e:
                print(f"[Server] PDF generation failed: {e}")

        return {
            "report": report_text,
            "ticker": ticker,
            "company_name": company_name,
            "intent": intent,
            "pdf_url": pdf_url,
        }
    else:
        # Standard query (chitchat / fast / research — no PDF)
        agent_out = run_agent(user_query)
        report_text = agent_out.get("report", "")
        ticker = agent_out.get("ticker", None)
        company_name = agent_out.get("company_name", None)
        intent = agent_out.get("intent", None)

        # Clean the output for non-PDF responses too
        if intent != "CHAT" and report_text:
            try:
                from tools.report_gen import clean_output
                report_text = clean_output(report_text, company_name)
            except Exception:
                pass

        return {
            "report": report_text,
            "ticker": ticker,
            "company_name": company_name,
            "intent": intent,
            "pdf_url": None,
        }


# ─── API Endpoints ──────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint — accepts a user message, processes it through
    the agent pipeline, and returns the response with an optional PDF URL.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # Run the blocking agent call in a thread pool
        result = await asyncio.to_thread(_run_agent_sync, message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"[Server] Error processing message: {e}")
        return ChatResponse(
            report="I encountered an issue while processing your request. Please try again.",
            intent="ERROR",
        )


@app.get("/api/download/{filename}")
async def download_pdf(filename: str):
    """Serve a generated PDF file for download."""
    # Sanitize filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUTS_DIR, safe_filename)

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="PDF not found.")

    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=safe_filename,
        headers={"Content-Disposition": f'inline; filename="{safe_filename}"'},
    )


@app.get("/")
async def serve_frontend():
    """Serve the chat frontend."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 60)
    print("  ARA-1 Financial Research Agent - Web Server")
    print("=" * 60)
    print()
    print("  Open in browser:  http://localhost:8899")
    print("  API docs:         http://localhost:8899/docs")
    print()
    print("=" * 60)
    print()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8899,
        reload=False,
        log_level="info",
    )
