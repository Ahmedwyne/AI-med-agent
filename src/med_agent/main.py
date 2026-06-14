from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import logging
import os
import re
import time
import threading
import uuid
import uvicorn

from med_agent.config.logging_config import configure_logging, request_id_var

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="MedAgent", description="Medical Research AI Agent")

# ── Request ID middleware (3.2) ───────────────────────────────────────────────
@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

# ── In-process metrics (3.3) ─────────────────────────────────────────────────
_metrics: dict = {"requests_total": 0, "errors_total": 0, "last_latency_ms": 0.0}
_metrics_lock = threading.Lock()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Mount static and templates for web UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> dict:
    with _metrics_lock:
        return dict(_metrics)

async def get_relevant_answer(user_query: str) -> str:
    from med_agent.crew import crew  # lazy import — env must be loaded first

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = crew.kickoff(inputs={"query": user_query})
            answer = result.raw if hasattr(result, "raw") else str(result)
            if not answer or not answer.strip():
                raise ValueError("The agent pipeline returned an empty response.")
            return answer.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "RateLimitError" in err:
                m = re.search(r"try again in ([0-9.]+)s", err)
                wait = float(m.group(1)) + 2 if m else 30
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit. Waiting {wait:.0f}s before retry {attempt+2}/{max_retries}")
                    await asyncio.sleep(wait)
                    continue
            raise

@app.post("/agent_query", response_model=QueryResponse)
async def agent_query(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()
    with _metrics_lock:
        _metrics["requests_total"] += 1
    try:
        answer = await get_relevant_answer(req.query)
        with _metrics_lock:
            _metrics["last_latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return QueryResponse(answer=answer)
    except Exception as e:
        with _metrics_lock:
            _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/ask")
async def ask_query(data: QueryRequest):
    t0 = time.monotonic()
    with _metrics_lock:
        _metrics["requests_total"] += 1
    try:
        answer = await get_relevant_answer(data.query)
        with _metrics_lock:
            _metrics["last_latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return JSONResponse({"result": answer})
    except Exception as e:
        with _metrics_lock:
            _metrics["errors_total"] += 1
        return JSONResponse({"error": str(e)}, status_code=500)


def run():
    """Entry point for `med_agent` CLI command."""
    uvicorn.run("med_agent.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
