import os
import re
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
        # Format result for better display: preserve line breaks and highlight PMIDs
        
        
        def highlight_pmids(text):
            # Make PMIDs bold and clickable to PubMed
            return re.sub(r'(PMID: ?(\d+))', r'<b><a href="https://pubmed.ncbi.nlm.nih.gov/\2/" target="_blank">\1</a></b>', text)
        formatted_result = highlight_pmids(result_text).replace('\n', '<br>')
        return JSONResponse({"result": formatted_result, "query_type": query_type})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
