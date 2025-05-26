from typing import Any
from crewai.tools import BaseTool

class MedicalTool(BaseTool):
    """Base class for all medical tools."""
    def _run(self, query: str, **kwargs: Any) -> dict:
        """All tools must implement this method."""
        raise NotImplementedError()
        
    async def _arun(self, query: str, **kwargs: Any) -> dict:
        """Optional async implementation."""
        raise NotImplementedError()
