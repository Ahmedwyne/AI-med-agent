from med_agent.tools.base import MedicalTool
from typing import List, Dict
import logging

class SynthesisTool(MedicalTool):
    """
    Synthesizes a structured, evidence-based answer from article summaries and other sources.
    """
    name: str = "SynthesisTool"
    description: str = "Synthesizes a concise, structured, and evidence-based answer to the user query using provided article summaries and other reputable sources."

    def _run(self, query: str, articles: List[Dict], other_sources: List[Dict] = None) -> str:
        """
        Args:
            query (str): The user's medical question.
            articles (List[Dict]): List of article summaries (from PubMed, etc.).
            other_sources (List[Dict], optional): List of summaries from other reputable sources.
        Returns:
            str: Structured, evidence-based answer with citations.
        """
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

        answer = f"**Question:** {query}\n\n"
        # Always format the answer in structured Markdown for better UI/UX
        answer_sections = []
        # Question section
        answer_sections.append(f"## Question\n{query}\n")
        # CDC Guidelines
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
        # Key Research Findings
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
        # Clinical Trials
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
        # Other reputable sources
        if other_sources:
            other = [src for src in other_sources if src.get('source') not in ['ClinicalTrials.gov', 'CDC']]
            if other:
                other_section = ["### Other Reputable Sources"]
                for src in other:
                    other_section.append(f"- {src.get('summary', 'No summary')} ({src.get('source', 'Unknown')})")
                answer_sections.append("\n".join(other_section))
        # Add a note
        answer_sections.append("\n**Note:** This answer is synthesized from the latest available evidence. Always consult a healthcare professional for medical advice.")
        return "\n\n".join(answer_sections)
