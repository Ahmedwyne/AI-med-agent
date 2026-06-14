from typing import Any
from crewai.tools import BaseTool
import requests
from requests.adapters import HTTPAdapter

# Shared session with connection pooling — reuses TCP/TLS connections across all tool calls.
http_session = requests.Session()
http_session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20))
http_session.mount("http://", HTTPAdapter(pool_connections=10, pool_maxsize=20))

class MedicalTool(BaseTool):
    """Base class for all medical tools."""
    def _run(self, query: str, **kwargs: Any) -> dict:
        """All tools must implement this method."""
        raise NotImplementedError()

    async def _arun(self, query: str, **kwargs: Any) -> dict:
        """Optional async implementation."""
        raise NotImplementedError()
