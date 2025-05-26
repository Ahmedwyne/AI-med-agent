from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from med_agent.tools.pubmed import PubMedSearchTool, PubMedFetchTool
from med_agent.tools.drugs import DrugInfoTool

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

pubmed_search = PubMedSearchTool()
pubmed_fetch = PubMedFetchTool()
drug_info = DrugInfoTool()

@app.post("/test_search")
async def test_search(data: QueryRequest):
    try:
        # Search PubMed
        search_result = pubmed_search._run(query=data.query)
        pmids = search_result.get("pmids", "")
        
        # Fetch abstracts if PMIDs found
        abstracts = ""
        if pmids:
            fetch_result = pubmed_fetch._run(pmids=pmids)
            abstracts = fetch_result.get("abstracts", "")
        
        # Get drug info for known drugs
        drug_results = {}
        for drug in ["aspirin", "warfarin", "ibuprofen"]:
            if drug.lower() in data.query.lower():
                drug_results[drug] = drug_info._run(query=drug)
        
        return JSONResponse({
            "pubmed_results": {
                "pmids": pmids,
                "abstracts": abstracts
            },
            "drug_info": drug_results
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
