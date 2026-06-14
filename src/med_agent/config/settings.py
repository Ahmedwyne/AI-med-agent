import os
from dotenv import load_dotenv, find_dotenv

# Load .env if present — not required in Docker (vars injected via env_file/environment)
_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    find_dotenv(usecwd=True),
    os.path.join(_here, "..", ".env"),
    os.path.join(_here, "..", "..", "..", ".env"),
]
dotenv_path = next((p for p in _candidates if p and os.path.isfile(p)), None)
if dotenv_path:
    load_dotenv(dotenv_path)

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 2048))

# Legacy Groq settings (kept for fallback / reference)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 1024))

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_DIR = os.getenv("VECTOR_INDEX_DIR", "med_agent/vector_store")

# ── External APIs ─────────────────────────────────────────────────────────────
DRUG_API_KEY  = os.getenv("DRUG_API_KEY")
NCBI_API_KEY  = os.getenv("NCBI_API_KEY")
NCBI_EMAIL    = os.getenv("NCBI_EMAIL", "")   # Required by NCBI ToS — set in .env
PUBMED_RETMAX = int(os.getenv("PUBMED_RETMAX", 5))

# ── Validation ────────────────────────────────────────────────────────────────
if not GEMINI_API_KEY and not GROQ_API_KEY:
    raise EnvironmentError(
        "No LLM API key found. Set GEMINI_API_KEY (recommended) or GROQ_API_KEY in .env"
    )

__all__ = [
    "GEMINI_API_KEY", "LLM_MODEL", "LLM_MAX_TOKENS",
    "GROQ_API_KEY", "GROQ_MODEL",
    "EMBED_MODEL", "VECTOR_DIR",
    "DRUG_API_KEY", "NCBI_API_KEY", "NCBI_EMAIL", "PUBMED_RETMAX",
]
