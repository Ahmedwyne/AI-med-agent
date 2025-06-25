from med_agent.agents.base import BaseAgent
from med_agent.tools.pubmed import PubMedSearch, PubMedFetch
from med_agent.tools.clinicaltrials import ClinicalTrialsSearch
from med_agent.tools.cdc import CDCGuidelines
from med_agent.tools.synthesis import EvidenceSynthesizer

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.tools = [
            PubMedSearch(),
            PubMedFetch(),
            ClinicalTrialsSearch(),
            CDCGuidelines(),
            EvidenceSynthesizer()
        ]

    async def process_task(self, task: dict) -> dict:
        query = task.get('query', '')
        focus_areas = task.get('focus_areas', [])
        
        # Step 1: PubMed Search
        pubmed_results = await self.tools[0]._arun(query=query)
        if pubmed_results.get('pmids'):
            article_details = await self.tools[1]._arun(pmids=pubmed_results['pmids'])
        else:
            article_details = []
        
        # Step 2: Clinical Trials Search
        trials = await self.tools[2]._arun(query=query)
        
        # Step 3: CDC Guidelines
        guidelines = await self.tools[3]._arun(query=query)
        
        # Step 4: Evidence Synthesis
        evidence = await self.tools[4]._arun(
            articles=article_details,
            other_sources=[*trials, *guidelines]
        )
        
        return {
            'research_findings': evidence,
            'articles': article_details,
            'trials': trials,
            'guidelines': guidelines
        }