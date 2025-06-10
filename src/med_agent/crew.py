import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from litellm import RateLimitError
from med_agent.tools.pubmed import PubMedSearchTool, PubMedFetchTool
from med_agent.config.settings import GROQ_MAX_TOKENS
from med_agent.tools.drugs import DrugInfoTool
from med_agent.tools.clinicaltrials import ClinicalTrialsGovTool
from med_agent.tools.cdc import CDCGuidelinesTool
from med_agent.agents.embedding_tasks import (
    EmbedAndIndexTool,
    RetrieveChunksTool,
    GenerateSummaryTool,
)

# Load environment variables
load_dotenv()

# Configure retry parameters
MAX_RETRIES = 5  
BASE_DELAY = 10.0  
TOKEN_LIMIT = GROQ_MAX_TOKENS  

def create_llm_with_retries():
    """Create LLM instance with retry logic for rate limits and force max_tokens limit."""
    import re
    class RetryLLM(LLM):
        def __init__(self):
            super().__init__(
                model="llama-3.3-70b-versatile",
                api_base="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY,
                temperature=0.7,
                max_tokens=TOKEN_LIMIT
            )
        def chat_completion(self, messages, *args, **kwargs):
            # Always enforce max_tokens limit
            kwargs["max_tokens"] = TOKEN_LIMIT
            # Optionally truncate messages if too long (simple check)
            total_content = " ".join(m.get("content", "") for m in messages)
            if len(total_content) > 6000:  # crude char-based limit
                print("[WARN] Prompt too long, truncating context for rate limit safety.")
                for m in messages:
                    if "content" in m:
                        m["content"] = m["content"][-4000:]  # keep last 4000 chars
            for attempt in range(MAX_RETRIES):
                try:
                    initial_delay = 5.0 if attempt == 0 else BASE_DELAY * (4 ** attempt)
                    print(f"\nWaiting {initial_delay}s before request (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(initial_delay)
                    response = super().chat_completion(messages, *args, **kwargs)
                    # Defensive: check for None or empty response
                    if not response or (isinstance(response, str) and not response.strip()):
                        print("[ERROR] LLM returned None or empty response. Returning fallback message.")
                        return "[LLM ERROR] No response generated. Please try again or check model/provider settings."
                    return response
                except RateLimitError as e:
                    error_msg = str(e)
                    wait_time = None
                    match = re.search(r"try again in ([0-9.]+)s", error_msg)
                    if match:
                        wait_time = float(match.group(1))
                        print(f"\nGroq rate limit: waiting {wait_time}s as suggested by API.")
                        time.sleep(wait_time)
                    else:
                        delay = BASE_DELAY * (4 ** attempt)
                        print(f"\nRate limit reached. Waiting {delay}s before retry")
                        time.sleep(delay)
                    if attempt < MAX_RETRIES - 1:
                        continue
                    raise
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_DELAY * (3 ** attempt)
                        print(f"\nError occurred. Waiting {delay}s before retry")
                        time.sleep(delay)
                        continue
                    raise
    return RetryLLM()

# Set up Groq LLM with retry logic
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_llm = create_llm_with_retries()

# Initialize tools
pubmed_search = PubMedSearchTool()
pubmed_fetch = PubMedFetchTool()
drug_info = DrugInfoTool()
clinical_trials = ClinicalTrialsGovTool()
cdc_guidelines = CDCGuidelinesTool()
embed_index = EmbedAndIndexTool()
retrieve_chunks = RetrieveChunksTool()
generate_summary = GenerateSummaryTool()

# Agent definitions with MCP integration
research_agent = Agent(
    role="Medical Literature Researcher",
    goal="Find and analyze relevant medical literature with citations",
    backstory="""I am an expert at searching and analyzing medical research papers. 
    I store my findings in MCP notes for other agents to use, and always include 
    PubMed IDs (PMIDs) as references.""",
    tools=[pubmed_search, pubmed_fetch, clinical_trials, cdc_guidelines],
    llm=groq_llm,
    allow_delegation=True,  
    verbose=True
)

drug_expert = Agent(
    role="Drug Information Expert",
    goal="Provide accurate drug information and analysis",
    backstory="""I specialize in pharmaceutical knowledge and drug interactions.
    I read research findings from MCP notes and contribute my analysis back to the shared context.""",
    tools=[drug_info],
    llm=groq_llm,
    allow_delegation=True,  
    verbose=True
)

synthesis_agent = Agent(
    role="Medical Information Synthesizer", 
    goal="Process and synthesize medical information",
    backstory="""I excel at understanding and summarizing complex medical information.
    I combine literature findings and drug analysis from MCP notes to create comprehensive summaries.""",
    tools=[embed_index, retrieve_chunks, generate_summary, clinical_trials, cdc_guidelines],
    llm=groq_llm,
    allow_delegation=True,  
    verbose=True
)

# Create medical crew with proper task definitions and MCP integration
crew = Crew(
    agents=[research_agent, drug_expert, synthesis_agent],
    tasks=[
        Task(
            description="""Search PubMed for the most recent and relevant articles about the user's query.
            Focus on high-quality clinical studies, meta-analyses, and systematic reviews.
            For each relevant article, fetch the abstract or full text (when available) and extract key findings, results, and conclusions directly related to the user's query.
            Use specific medical terms and boolean operators for precise search.
            Must include publication dates, impact factors, and PMIDs when available.""",
            agent=research_agent,
            expected_output="A curated list of relevant PubMed articles with extracted key findings, results, and conclusions, including publication details and PMIDs",
            context_format="Medical query: {query}\nRequired information: Recent studies, clinical evidence, safety data",
            input_keys=["query"],
        ),
        Task(
            description="""Analyze drug interactions, mechanisms, and safety profiles.
            Extract specific dosing concerns, contraindications, and risk factors.
            Prioritize evidence from clinical guidelines and FDA recommendations.
            Cross-reference drug information with the research findings.""",
            agent=drug_expert,
            expected_output="Detailed analysis of drug interactions, mechanisms, and safety considerations with evidence levels"
        ),
        Task(
            description="""Generate an evidence-based clinical summary that directly answers the user's medical query using the extracted findings from the relevant studies.
            The summary must:
            1. Synthesize key findings, results, and conclusions from the fetched studies
            2. Address the user's specific question with accurate, up-to-date information
            3. Include key drug interactions and mechanisms, risk stratification, patient factors, and monitoring recommendations if relevant
            4. Cite specific PMIDs and evidence levels for each claim or recommendation
            5. Present references as a list of PMIDs at the end of the summary.""",
            agent=synthesis_agent,
            expected_output="Structured, accurate clinical summary that answers the user's query, with evidence grades and PMID citations"
        )
    ],
    verbose=True,
    processes={
        "medical-agent-mcp": {
            "command": "medical-agent-mcp",
            "type": "stdio"
        }
    }
)