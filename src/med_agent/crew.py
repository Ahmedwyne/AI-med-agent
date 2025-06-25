import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from litellm import RateLimitError
from med_agent.tools.pubmed import PubMedSearch, PubMedFetch
from med_agent.config.settings import GROQ_MAX_TOKENS
from med_agent.tools.drugs import DrugInfoTool
from med_agent.tools.clinicaltrials import ClinicalTrialsSearch
from med_agent.tools.cdc import CDCGuidelines
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
pubmed_search = PubMedSearch()
pubmed_fetch = PubMedFetch()
drug_info = DrugInfoTool()
clinical_trials = ClinicalTrialsSearch()
cdc_guidelines = CDCGuidelines()
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
    I always prioritize real-time evidence from PubMed, CDC, and ClinicalTrials.gov, drug database and only use local vector store data if no relevant live evidence is found.""",
    tools=[pubmed_search, pubmed_fetch, clinical_trials, cdc_guidelines, retrieve_chunks, generate_summary, embed_index],
    llm=groq_llm,
    allow_delegation=True,  
    verbose=True
)

# Create medical crew with proper task definitions and MCP integration
crew = Crew(
    agents=[research_agent, drug_expert, synthesis_agent],
    tasks=[
        Task(
            description="""Research Phase:
            1. Search PubMed for the most recent (last 2 years) and relevant articles about the user's query
            2. Prioritize: meta-analyses > systematic reviews > RCTs > observational studies
            3. For each article:
                - Extract and grade evidence (A: High, B: Moderate, C: Low)
                - Document: PMID, publication date, journal impact factor, study type
                - Summarize key findings, methodology, and limitations
            4. Search ClinicalTrials.gov for ongoing relevant trials
            5. Check CDC guidelines for current recommendations
            Store all findings in MCP notes with proper citation formatting.""",
            agent=research_agent,
            expected_output="""Structured research findings with:
            1. Evidence table (Study details, PMID, evidence grade)
            2. Key findings by evidence level
            3. Ongoing clinical trials summary
            4. Current guideline recommendations""",
            context_format="Medical query: {query}\nRequired focus: {focus_areas}",
            input_keys=["query", "focus_areas"],
        ),
        Task(
            description="""Drug Analysis Phase:
            1. Review research findings from MCP notes
            2. For each relevant drug/therapy:
                - Document mechanism of action
                - Analyze safety profile and contraindications
                - Review drug-drug interactions
                - Extract specific dosing guidelines
                - Note monitoring requirements
            3. Cross-reference findings with:
                - FDA recommendations
                - Current clinical guidelines
                - Phase III/IV trial data
            4. Grade recommendations (Class I, IIa, IIb, III)""",
            agent=drug_expert,
            expected_output="""Detailed drug analysis with:
            1. Drug mechanisms and interactions
            2. Safety profiles with evidence grades
            3. Monitoring recommendations
            4. Recommendation classification""",
        ),
        Task(
            description="""Synthesis Phase:
            1. Integrate findings from research and drug analysis phases
            2. Generate evidence-based summary addressing:
                - Primary clinical question
                - Current evidence overview
                - Specific recommendations with grades
                - Safety considerations
                - Monitoring requirements
            3. Format output with:
                - Clear evidence levels for each statement
                - In-text PMID citations
                - Organized sections with headers
                - Clinical pearls and key warnings
                - Reference list with PMIDs and evidence grades""",
            agent=synthesis_agent,
            expected_output="""Comprehensive clinical summary with:
            1. Evidence-based answers to query
            2. Graded recommendations
            3. Safety and monitoring guidance
            4. Properly cited references""",
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