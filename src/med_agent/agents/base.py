from crewai import Agent
from typing import List, Dict, Any
from med_agent.tools.base import MedicalTool

class BaseAgent(Agent):
    """Base agent class for all medical agents"""
    
    def __init__(self):
        self.tools: List[MedicalTool] = []
        super().__init__(
            role="Medical Research Agent",
            goal="Process medical information and provide evidence-based answers",
            backstory="An AI agent specialized in medical research and evidence synthesis",
            tools=self.tools
        )
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        raise NotImplementedError()
