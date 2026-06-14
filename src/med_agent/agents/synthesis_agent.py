from med_agent.agents.base import BaseAgent
from med_agent.tools.synthesis import (
    EvidenceSynthesizer,
    RecommendationGenerator,
    OutputFormatter,
    CitationManager,
)


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            tools=[
                EvidenceSynthesizer(),
                RecommendationGenerator(),
                OutputFormatter(),
                CitationManager(),
            ],
            role="Medical Information Synthesizer",
            goal="Integrate and synthesize medical evidence into clear clinical summaries",
            backstory="An AI agent that combines research findings and drug analysis into evidence-based recommendations",
        )

    async def process_task(self, task: dict) -> dict:
        research_findings = task.get('research_findings', {})
        drug_analysis = task.get('drug_analysis', {})

        integrated_evidence = self._integrate_evidence(research_findings, drug_analysis)
        recommendations = self._generate_recommendations(integrated_evidence)
        citations = self._collect_citations(research_findings)
        final_output = self._format_output(integrated_evidence, recommendations, citations)
        return final_output

    def _integrate_evidence(self, research: dict, drug_analysis: dict) -> dict:
        """Organize evidence by grade from research findings and drug analysis."""
        evidence = {'high_grade': [], 'moderate_grade': [], 'low_grade': [], 'ungraded': []}

        evidence_table = research.get('evidence_table', {})
        if isinstance(evidence_table, dict):
            for grade_key, items in evidence_table.items():
                if grade_key in evidence:
                    evidence[grade_key].extend(items)
        elif isinstance(evidence_table, list):
            for finding in evidence_table:
                grade = finding.get('evidence_grade', '').upper()
                if grade == 'A':
                    evidence['high_grade'].append(finding)
                elif grade == 'B':
                    evidence['moderate_grade'].append(finding)
                elif grade == 'C':
                    evidence['low_grade'].append(finding)
                else:
                    evidence['ungraded'].append(finding)

        for drug in drug_analysis.get('recommendations', []):
            raw_grade = drug.get('grade', '').lower()
            key = f"{raw_grade}_grade"
            if key in evidence:
                evidence[key].append(drug)
            else:
                evidence['ungraded'].append(drug)

        return evidence

    def _generate_recommendations(self, evidence: dict) -> list:
        recommendations = []
        high = evidence.get('high_grade', [])
        if high:
            recommendations.append({
                'grade': 'A',
                'text': f"Strong evidence from {len(high)} high-grade source(s) supports this intervention.",
                'sources': [item.get('pmid', '') for item in high if item.get('pmid')],
            })
        moderate = evidence.get('moderate_grade', [])
        if moderate:
            recommendations.append({
                'grade': 'B',
                'text': f"Moderate evidence from {len(moderate)} source(s) supports this intervention.",
                'sources': [item.get('pmid', '') for item in moderate if item.get('pmid')],
            })
        if not recommendations:
            recommendations.append({
                'grade': 'C',
                'text': 'Insufficient high-quality evidence found. Consult a healthcare professional.',
                'sources': [],
            })
        return recommendations

    def _collect_citations(self, research: dict) -> list:
        citations = research.get('citations', [])
        if not citations:
            abstracts = research.get('abstracts', [])
            citations = [f"PMID: {a['pmid']}" for a in abstracts if isinstance(a, dict) and a.get('pmid')]
        return citations

    def _format_output(self, evidence: dict, recommendations: list, citations: list) -> dict:
        return {
            'clinical_answer': {
                'summary': self._summarize(evidence),
                'recommendations': recommendations,
                'evidence_levels': {k: len(v) for k, v in evidence.items()},
            },
            'evidence_details': {
                grade: self._format_evidence_section(items)
                for grade, items in evidence.items()
            },
            'references': citations,
        }

    def _summarize(self, evidence: dict) -> str:
        total = sum(len(v) for v in evidence.values())
        high = len(evidence.get('high_grade', []))
        return (
            f"Synthesis based on {total} evidence item(s), "
            f"including {high} high-grade source(s)."
        )

    def _format_evidence_section(self, evidence_list: list) -> list:
        formatted = []
        for item in evidence_list:
            if not isinstance(item, dict):
                continue
            formatted.append({
                'finding': item.get('finding', item.get('text', str(item))),
                'citation': f"[PMID: {item['pmid']}]" if item.get('pmid') else '',
                'grade': item.get('evidence_grade', item.get('grade', '')),
                'implications': item.get('clinical_implications', ''),
            })
        return formatted
