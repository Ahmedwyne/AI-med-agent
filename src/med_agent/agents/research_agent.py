from med_agent.agents.base import BaseAgent
from med_agent.tools.pubmed import PubMedSearch, PubMedFetch
from med_agent.tools.clinicaltrials import ClinicalTrialsSearch
from med_agent.tools.cdc import CDCGuidelines
from med_agent.tools.synthesis import EvidenceSynthesizer


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            tools=[
                PubMedSearch(),
                PubMedFetch(),
                ClinicalTrialsSearch(),
                CDCGuidelines(),
                EvidenceSynthesizer(),
            ],
            role="Medical Research Agent",
            goal="Search and synthesize medical evidence from PubMed, ClinicalTrials.gov, and CDC guidelines",
            backstory="An AI agent specialized in biomedical literature search and evidence grading",
        )

    async def process_task(self, task: dict) -> dict:
        query = task.get('query', '')

        pubmed_search = self.tools[0]
        pubmed_fetch = self.tools[1]
        clinical_trials = self.tools[2]
        cdc_guidelines = self.tools[3]

        pubmed_results = pubmed_search._run(query=query)
        pmids = pubmed_results.get('pmids', [])
        article_details = pubmed_fetch._run(pmids=pmids) if pmids else {"abstracts": []}

        trials = clinical_trials._run(query=query)
        guidelines = cdc_guidelines._run(query=query)

        return {
            'research_findings': article_details,
            'articles': article_details.get('abstracts', []),
            'trials': trials,
            'guidelines': guidelines,
        }
