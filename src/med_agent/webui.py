import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from med_agent.crew import research_agent, synthesis_agent, drug_expert, Task
from med_agent.tools.query_classification import classify_query_type

app = FastAPI()

# Set up static and template directories
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
def ask_query(request: Request, data: QueryRequest):
    user_query = data.query
    query_type = classify_query_type(user_query)
    print(f"[INFO] Query classified as: {query_type}")


    # Route to the most relevant agent based on query type
    if query_type == "drug_info":
        agent = drug_expert
    elif query_type in ["treatment", "prevention"]:
        agent = synthesis_agent
    else:
        agent = research_agent
    task = Task(
        description=f"({query_type.upper()}) {user_query}",
        agent=agent,
        expected_output="A detailed summary with PMIDs and reputable source references"
    )
    try:
        result = agent.execute_task(task)
        # If result is a dict with 'result' key, extract it
        if isinstance(result, dict) and 'result' in result:
            result_text = result['result']
        else:
            result_text = result
        # Do not replace newlines with <br> or do any HTML conversion here.
        # The frontend will render Markdown using marked.js for best UI/UX.
        return JSONResponse({"result": result_text, "query_type": query_type})
    
    except Exception as e:
        # Friendly error message for LLM/Groq errors and rate limits
        err_msg = str(e)
        if (
            'GroqException' in err_msg or 'APIError' in err_msg or 'Internal Server Error' in err_msg or
            'rate_limit_exceeded' in err_msg or 'RateLimitError' in err_msg or 'rate limit reached' in err_msg
        ):
            user_msg = (
                "<b>Sorry, the AI service is temporarily unavailable or rate-limited.</b><br>"
                "You have reached the daily or per-minute usage limit for the current model.<br>"
                "Please try again later, reduce the length of your query, or consider upgrading your service tier if you need more usage.<br>"
                "<br><small>Provider info: <a href='https://docs.litellm.ai/docs/providers' target='_blank'>Provider List</a></small>"
            )
            return JSONResponse({"error": user_msg}, status_code=503)
        return JSONResponse({"error": err_msg}, status_code=500)
