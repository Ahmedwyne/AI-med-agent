FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Dependency layer (cached unless pyproject.toml changes) ---
COPY pyproject.toml README.md ./
# Minimal stub so hatchling can build the package metadata
RUN mkdir -p src/med_agent && touch src/med_agent/__init__.py

# Install core deps first (crewai pulls a lot of transitive deps)
RUN uv pip install --system --no-cache \
    crewai>=0.119.0 \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    httpx>=0.27.0 \
    python-dotenv>=1.0.0 \
    "pydantic>=2.0.0" \
    "requests>=2.31.0" \
    beautifulsoup4>=4.12.0 \
    "groq>=0.9.0" \
    "litellm>=1.40.0" \
    "jinja2>=3.1.0" \
    python-multipart>=0.0.9

# Install google-genai in a separate layer (large package, install alone to avoid OOM)
RUN uv pip install --system --no-cache "google-genai>=1.0.0"

# --- Application layer ---
COPY src/ src/
COPY knowledge/ knowledge/

# Install the med_agent package itself (no extra deps)
RUN uv pip install --system --no-cache --no-deps -e .

# Vector store directory (overridden by volume mount at runtime)
RUN mkdir -p med_agent/vector_store

EXPOSE 8000

CMD ["uvicorn", "med_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
