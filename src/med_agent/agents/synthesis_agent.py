from med_agent.agents.base import BaseAgent
from med_agent.tools.synthesis import (
    EvidenceSynthesizer,
    RecommendationGenerator, 
    OutputFormatter,
    CitationManager
)

class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.tools = [
            EvidenceSynthesizer(),
            RecommendationGenerator(),
            OutputFormatter(),
            CitationManager()
        ]
        
    async def process_task(self, task: dict) -> dict:
        # Get research and drug analysis from MCP notes
        research_findings = self.get_mcp_notes('research_findings')
        drug_analysis = self.get_mcp_notes('drug_analysis')
        
        # Step 1: Evidence Integration
        integrated_evidence = await self.integrate_evidence(
            research_findings,
            drug_analysis
        )
        
        # Step 2: Generate Recommendations
        recommendations = await self.generate_recommendations(integrated_evidence)
        
        # Step 3: Format Final Output
        final_output = self.format_output(
            integrated_evidence,
            recommendations,
            research_findings['citations']
        )
        
        return final_output
        
    async def integrate_evidence(self, research, drug_analysis):
        """Integrate and organize evidence from multiple sources"""
        evidence = {
            'high_grade': [],
            'moderate_grade': [],
            'low_grade': []
        }
        
        # Organize evidence by grade
        for finding in research['evidence_table']:
            if finding['evidence_grade'] == 'A':
                evidence['high_grade'].append(finding)
            elif finding['evidence_grade'] == 'B':
                evidence['moderate_grade'].append(finding)
            else:
                evidence['low_grade'].append(finding)
                
        # Add drug analysis with appropriate grading
        for drug in drug_analysis['recommendations']:
            evidence[f"{drug['grade'].lower()}_grade"].append(drug)
            
        return evidence
        
    def format_output(self, evidence, recommendations, citations):
        """Format final output with proper structure and citations"""
        output = {
            'clinical_answer': {
                'summary': self.generate_summary(evidence),
                'recommendations': recommendations,
                'evidence_levels': self.summarize_evidence_levels(evidence),
                'safety_monitoring': self.extract_safety_info(evidence)
            },
            'evidence_details': {
                'high_grade': self.format_evidence_section(evidence['high_grade']),
                'moderate_grade': self.format_evidence_section(evidence['moderate_grade']),
                'low_grade': self.format_evidence_section(evidence['low_grade'])
            },
            'references': citations
        }
        
        return output
        
    def format_evidence_section(self, evidence_list):
        """Format evidence with inline citations and clear structure"""
        formatted = []
        for item in evidence_list:
            entry = {
                'finding': item['finding'],
                'citation': f"[PMID: {item['pmid']}]",
                'grade': item['evidence_grade'],
                'implications': item.get('clinical_implications', '')
            }
            formatted.append(entry)
        return formatted