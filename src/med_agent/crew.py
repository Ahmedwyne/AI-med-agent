import litellm
litellm.drop_params = True
litellm.num_retries = 3  # auto-retry on rate limit / transient errors

from crewai import Agent, Task, Crew, Process, LLM
from med_agent.tools.pubmed import PubMedSearch, PubMedFetch
from med_agent.tools.drugs import DrugInfoTool
from med_agent.tools.clinicaltrials import ClinicalTrialsSearch
from med_agent.tools.cdc import CDCGuidelines
from med_agent.agents.embedding_tasks import (
    EmbedAndIndexTool,
    RetrieveChunksTool,
    GenerateSummaryTool,
)
import os
from med_agent.config.settings import GEMINI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS

# ── LLM ──────────────────────────────────────────────────────────────────────
# gemini/gemini-2.0-flash: 1M TPM free tier — no rate limit issues.
# crewAI's native Gemini provider reads GOOGLE_API_KEY from the environment.
os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY or "")

llm = LLM(
    model=LLM_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.7,
    max_tokens=LLM_MAX_TOKENS,
)

# ── Tools ────────────────────────────────────────────────────────────────────
pubmed_search    = PubMedSearch()
pubmed_fetch     = PubMedFetch()
drug_info        = DrugInfoTool()
clinical_trials  = ClinicalTrialsSearch()
cdc_guidelines   = CDCGuidelines()
embed_index      = EmbedAndIndexTool()
retrieve_chunks  = RetrieveChunksTool()
generate_summary = GenerateSummaryTool()

# ── Agents ───────────────────────────────────────────────────────────────────
research_agent = Agent(
    role="Medical Literature Researcher",
    goal="Find and analyze relevant medical literature with citations",
    backstory=(
        "I am an expert at searching and analyzing medical research papers. "
        "I always include PubMed IDs (PMIDs) as references."
    ),
    tools=[pubmed_search, pubmed_fetch, clinical_trials, cdc_guidelines],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

drug_expert = Agent(
    role="Drug Information Expert",
    goal="Provide accurate drug information and analysis",
    backstory=(
        "I specialize in pharmaceutical knowledge and drug interactions. "
        "I cross-reference FDA recommendations and clinical guidelines."
    ),
    tools=[drug_info],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

synthesis_agent = Agent(
    role="Medical Information Synthesizer",
    goal="Synthesize medical information into a clear, cited summary",
    backstory=(
        "I integrate evidence from PubMed, CDC, and ClinicalTrials.gov into "
        "concise, graded clinical summaries with proper citations."
    ),
    tools=[pubmed_search, pubmed_fetch, clinical_trials, cdc_guidelines,
           retrieve_chunks, generate_summary, embed_index],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# ── Tasks ────────────────────────────────────────────────────────────────────
research_task = Task(
    description=(
        "Research Phase for query: {query}\n"
        "1. Search PubMed for recent, relevant articles (last 2 years preferred).\n"
        "2. Prioritize: meta-analyses > systematic reviews > RCTs.\n"
        "3. For each article note PMID, study type, and key findings.\n"
        "4. Search ClinicalTrials.gov for ongoing trials.\n"
        "5. Check CDC guidelines for current recommendations."
    ),
    agent=research_agent,
    expected_output=(
        "Structured findings: evidence table with PMID and evidence grade, "
        "key findings, ongoing trials, guideline recommendations."
    ),
)

drug_task = Task(
    description=(
        "Drug Analysis Phase:\n"
        "Review the research findings and for each relevant drug/therapy document: "
        "mechanism of action, safety profile, contraindications, drug interactions, "
        "dosing guidelines, and monitoring requirements. "
        "Grade recommendations (Class I, IIa, IIb, III)."
    ),
    agent=drug_expert,
    expected_output=(
        "Drug analysis: mechanisms, interactions, safety profiles with evidence grades, "
        "monitoring recommendations, recommendation classification."
    ),
    context=[research_task],
)

synthesis_task = Task(
    description=(
        "Synthesis Phase:\n"
        "Integrate the research and drug analysis into a comprehensive clinical summary. "
        "Include: evidence-based answer, graded recommendations, safety considerations, "
        "monitoring requirements, and a reference list with PMIDs."
    ),
    agent=synthesis_agent,
    expected_output=(
        "Comprehensive clinical summary with evidence-based answers, "
        "graded recommendations, safety/monitoring guidance, and cited references."
    ),
    context=[research_task, drug_task],
)

# ── Crew ─────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[research_agent, drug_expert, synthesis_agent],
    tasks=[research_task, drug_task, synthesis_task],
    process=Process.sequential,
    verbose=True,
)
