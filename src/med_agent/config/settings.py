import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError(".env file not found. Please create one in the project root.")
load_dotenv(dotenv_path)

# Groq LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", 5))  
GROQ_RETRY_DELAY = float(os.getenv("GROQ_RETRY_DELAY", 10.0))  
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS"))  # reduced from 2048 to stay within stricter limits
GROQ_RATE_LIMIT_TPM = int(os.getenv("GROQ_RATE_LIMIT_TPM", 12000))  # tokens per minute limit

# Embedding model for SentenceTransformer
EMBED_MODEL  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Vector store settings
VECTOR_DIR   = os.getenv("VECTOR_INDEX_DIR", "med_agent/vector_store")

# Drug API key (for RxNorm )
DRUG_API_KEY = os.getenv("DRUG_API_KEY")

# PubMed settings
PUBMED_RETMAX = int(os.getenv("PUBMED_RETMAX", 5))
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Validate required variables
required_vars = {"GROQ_API_KEY": GROQ_API_KEY, "EMBED_MODEL": EMBED_MODEL}
missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

__all__ = [
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "EMBED_MODEL",
    "VECTOR_DIR",
    "DRUG_API_KEY",
    "PUBMED_RETMAX",
]