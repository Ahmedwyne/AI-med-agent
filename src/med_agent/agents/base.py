from crewai import Agent
from typing import List, Dict, Any
from med_agent.tools.base import MedicalTool


class BaseAgent(Agent):
    """Base agent class for all medical agents"""

    def __init__(self, tools: List[MedicalTool] = None, **kwargs):
        super().__init__(
            role=kwargs.get("role", "Medical Research Agent"),
            goal=kwargs.get("goal", "Process medical information and provide evidence-based answers"),
            backstory=kwargs.get("backstory", "An AI agent specialized in medical research and evidence synthesis"),
            tools=tools or [],
            **{k: v for k, v in kwargs.items() if k not in ("role", "goal", "backstory")}
        )

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        raise NotImplementedError()
