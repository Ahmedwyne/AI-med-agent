import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.med_agent.tools.pubmed import PubMedSearchTool, PubMedFetchTool
from src.med_agent.tools.drugs import DrugInfoTool

def test_pubmed_search():
    """Test PubMed search functionality"""
    search_tool = PubMedSearchTool()
    query = "aspirin warfarin interaction"
    result = search_tool._run(query=query)
    print("\nPubMed Search Results:")
    print(result)
    return result

def test_drug_info():
    """Test drug information lookup"""
    drug_tool = DrugInfoTool()
    drugs = ["aspirin", "warfarin"]
    results = []
    print("\nDrug Information Results:")
    for drug in drugs:
        result = drug_tool._run(query=drug)
        print(f"\n{drug}: {result}")
        results.append(result)
    return results

if __name__ == "__main__":
    print("Testing PubMed Search...")
    pmids = test_pubmed_search()
    
    print("\nTesting Drug Information...")
    drug_info = test_drug_info()
