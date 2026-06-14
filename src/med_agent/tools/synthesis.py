from med_agent.tools.base import MedicalTool
from typing import List, Dict
import logging


class EvidenceSynthesizer(MedicalTool):
    name: str = "EvidenceSynthesizer"
    description: str = "Synthesizes and grades medical evidence from multiple sources"

    def _run(self, articles: List[Dict] = None, other_sources: List[Dict] = None) -> Dict:
        articles = articles or []
        other_sources = other_sources or []
        evidence = {"high_grade": [], "moderate_grade": [], "low_grade": [], "ungraded": []}
        for item in articles:
            grade = item.get("evidence_grade", "").upper()
            if grade == "A":
                evidence["high_grade"].append(item)
            elif grade == "B":
                evidence["moderate_grade"].append(item)
            elif grade == "C":
                evidence["low_grade"].append(item)
            else:
                evidence["ungraded"].append(item)
        return {"evidence_table": evidence, "sources": other_sources}


class RecommendationGenerator(MedicalTool):
    name: str = "RecommendationGenerator"
    description: str = "Generates clinical recommendations based on evidence"

    def _run(self, evidence: Dict = None) -> List[Dict]:
        evidence = evidence or {}
        recommendations = []
        high = evidence.get("high_grade", [])
        if high:
            recommendations.append({
                "grade": "A",
                "recommendation": f"Strong evidence from {len(high)} high-grade source(s) supports this intervention.",
                "sources": [item.get("pmid", "") for item in high if item.get("pmid")]
            })
        moderate = evidence.get("moderate_grade", [])
        if moderate:
            recommendations.append({
                "grade": "B",
                "recommendation": f"Moderate evidence from {len(moderate)} source(s) supports this intervention.",
                "sources": [item.get("pmid", "") for item in moderate if item.get("pmid")]
            })
        if not recommendations:
            recommendations.append({
                "grade": "C",
                "recommendation": "Insufficient high-quality evidence found. Consult a healthcare professional.",
                "sources": []
            })
        return recommendations


class OutputFormatter(MedicalTool):
    name: str = "OutputFormatter"
    description: str = "Formats medical evidence and recommendations into structured output"

    def _run(self, evidence: Dict = None, recommendations: List[Dict] = None, citations: List = None) -> str:
        evidence = evidence or {}
        recommendations = recommendations or []
        citations = citations or []
        lines = ["## Clinical Summary\n"]
        for rec in recommendations:
            lines.append(f"**Grade {rec.get('grade', '?')}:** {rec.get('recommendation', '')}")
        lines.append("\n## Evidence Levels")
        for grade, items in evidence.items():
            if items:
                lines.append(f"- **{grade.replace('_', ' ').title()}**: {len(items)} source(s)")
        if citations:
            lines.append("\n## References")
            for cite in citations:
                lines.append(f"- {cite}")
        return "\n".join(lines)


class CitationManager(MedicalTool):
    name: str = "CitationManager"
    description: str = "Manages and formats medical citations"

    def _run(self, query: str = "", articles: List[Dict] = None, other_sources: List[Dict] = None) -> str:
        articles = articles or []
        other_sources = other_sources or []
        logging.debug(f"Synthesizing answer for query: {query}")
        if not articles and not other_sources:
            answer = f"**Question:** {query}\n\n"
            answer += "---\n### Evidence Search Results\n- No relevant guidelines, articles, or clinical trials were found for this query in PubMed, CDC, or ClinicalTrials.gov.\n\n"
            answer += "**Suggestions:**\n"
            answer += "- Try searching with broader terms (e.g., 'HFpEF guidelines' or 'heart failure management guidelines').\n"
            answer += "- Visit [ESC Guidelines](https://www.escardio.org/Guidelines) or [ACC/AHA/HFSA Guidelines](https://www.acc.org/guidelines) for the latest updates.\n"
            answer += "- Consult a cardiologist for expert interpretation.\n\n"
            answer += "**Note:** The agent did not fabricate an answer, maintaining clinical safety and transparency."
            return answer

        answer_sections = [f"## Question\n{query}\n"]
        if other_sources:
            cdc = [src for src in other_sources if src.get('source') == 'CDC']
            if cdc:
                cdc_section = ["### CDC Guidelines"]
                for guideline in cdc:
                    summary = guideline.get('summary', 'No summary')
                    link = guideline.get('link')
                    if summary:
                        cdc_section.append(f"- {summary}")
                    if link:
                        cdc_section.append(f"[CDC Guidance Link]({link})")
                answer_sections.append("\n".join(cdc_section))

        if articles:
            findings = ["### Key Research Findings"]
            for idx, art in enumerate(articles, 1):
                title = art.get('title', 'No title')
                summary = art.get('clinical_summary', art.get('summary', 'No summary'))
                pmid = art.get('pmid')
                level = art.get('evidence_level', '')
                finding = f"{idx}. **{title}**"
                if level:
                    finding += f" ({level})"
                finding += f": {summary}"
                if pmid:
                    finding += f"  [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                findings.append(finding)
            answer_sections.append("\n".join(findings))

        if other_sources:
            ctgov = [src for src in other_sources if src.get('source') == 'ClinicalTrials.gov']
            if ctgov:
                trials = ["### Relevant Clinical Trials (ClinicalTrials.gov)"]
                for trial in ctgov:
                    nct = trial.get('nct')
                    title = trial.get('title', 'No title')
                    status = trial.get('status', 'Unknown status')
                    summary = trial.get('summary', 'No summary')
                    if nct:
                        trials.append(f"- **{title}** (Status: {status}) [NCT:{nct}](https://clinicaltrials.gov/ct2/show/{nct})\n  {summary}")
                    else:
                        trials.append(f"- **{title}** (Status: {status})\n  {summary}")
                answer_sections.append("\n".join(trials))

        answer_sections.append("\n**Note:** This answer is synthesized from the latest available evidence. Always consult a healthcare professional for medical advice.")
        return "\n\n".join(answer_sections)
