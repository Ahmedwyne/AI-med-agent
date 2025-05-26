import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from litellm import completion, RateLimitError

# Load environment variables from .env file
load_dotenv()

# Import your tool implementations
from med_agent.tools.pubmed import PubMedSearchTool, PubMedFetchTool
from med_agent.tools.drugs import DrugInfoTool
from med_agent.agents.embedding_tasks import (
    EmbedAndIndexTool,
    RetrieveChunksTool,
    GenerateSummaryTool,
)

# Configure retry parameters
MAX_RETRIES = 5  # Increased retries
BASE_DELAY = 10.0  # Increased base delay
TOKEN_LIMIT = 800  # Lower token limit to stay under TPM

def create_llm_with_retries():
    """Create LLM instance with retry logic for rate limits."""
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
            for attempt in range(MAX_RETRIES):
                try:
                    # Always add a small delay before each attempt
                    initial_delay = 5.0 if attempt == 0 else BASE_DELAY * (4 ** attempt)
                    print(f"\nWaiting {initial_delay}s before request (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(initial_delay)
                    
                    return super().chat_completion(messages, *args, **kwargs)
                except RateLimitError as e:
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_DELAY * (4 ** attempt)  # More aggressive backoff
                        print(f"\nRate limit reached. Waiting {delay}s before retry")
                        time.sleep(delay)
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
    tools=[pubmed_search, pubmed_fetch],
    llm=groq_llm,
    allow_delegation=True,  # Enable MCP delegation
    verbose=True
)

drug_expert = Agent(
    role="Drug Information Expert",
    goal="Provide accurate drug information and analysis",
    backstory="""I specialize in pharmaceutical knowledge and drug interactions.
    I read research findings from MCP notes and contribute my analysis back to the shared context.""",
    tools=[drug_info],
    llm=groq_llm,
    allow_delegation=True,  # Enable MCP delegation
    verbose=True
)

synthesis_agent = Agent(
    role="Medical Information Synthesizer", 
    goal="Process and synthesize medical information",
    backstory="""I excel at understanding and summarizing complex medical information.
    I combine literature findings and drug analysis from MCP notes to create comprehensive summaries.""",
    tools=[embed_index, retrieve_chunks, generate_summary],
    llm=groq_llm,
    allow_delegation=True,  # Enable MCP delegation
    verbose=True
)

# Create medical crew with proper task definitions and MCP integration
crew = Crew(
    agents=[research_agent, drug_expert, synthesis_agent],
    tasks=[
        Task(
            description="""Search PubMed for the most recent and relevant articles about the user's query.
            Focus on high-quality clinical studies, meta-analyses, and systematic reviews.
            Use specific medical terms and boolean operators for precise search.
            Must include publication dates and impact factors when available.""",
            agent=research_agent,
            expected_output="A curated list of relevant PubMed articles with brief summaries and publication details",
            context_format="Medical query: {query}\nRequired information: Recent studies, clinical evidence, safety data"
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
            description="""Generate an evidence-based clinical summary including:
            1. Key drug interactions and mechanisms
            2. Risk stratification and patient factors
            3. Monitoring recommendations
            4. Alternative approaches if applicable
            Must cite specific PMIDs and evidence levels for each claim.""",
            agent=synthesis_agent,
            expected_output="Structured clinical summary with evidence grades and citations"
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