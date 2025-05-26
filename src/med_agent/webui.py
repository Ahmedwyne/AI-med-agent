import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from med_agent.crew import research_agent, Task

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
    task = Task(
        description=f"{user_query}",
        agent=research_agent,
        expected_output="A detailed summary with PMIDs as references"
    )
    try:
        result = research_agent.execute_task(task)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
