from med_agent.tools.base import MedicalTool, http_session
from med_agent.tools.cache import get_cached
from typing import List, Dict
import logging
import urllib.parse


class ClinicalTrialsSearch(MedicalTool):
    """
    Fetches and summarizes relevant clinical trial evidence from ClinicalTrials.gov for a given query.
    Uses the v2 API (https://clinicaltrials.gov/api/v2/studies).
    """
    name: str = "ClinicalTrialsSearch"
    description: str = "Search ClinicalTrials.gov for relevant trials"

    def _run(self, query: str, max_results: int = 3) -> str:
        return get_cached(f"ClinicalTrials:{query}:{max_results}", lambda: self._fetch(query, max_results), ttl=3600)

    def _fetch(self, query: str, max_results: int) -> str:
        logging.debug(f"Searching ClinicalTrials.gov for: {query}")
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,BriefSummary,Condition,InterventionName,StartDate,CompletionDate,Phase",
        }
        try:
            resp = http_session.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("studies", [])
            results = []
            for study in studies:
                proto = study.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                desc_mod = proto.get("descriptionModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                design_mod = proto.get("designModule", {})
                conditions_mod = proto.get("conditionsModule", {})

                nct = id_mod.get("nctId", "N/A")
                title = id_mod.get("briefTitle", "No title")
                status = status_mod.get("overallStatus", "Unknown")
                summary = desc_mod.get("briefSummary", "No summary available.")
                conditions = ", ".join(conditions_mod.get("conditions", []))
                interventions = ", ".join(
                    i.get("name", "") for i in arms_mod.get("interventions", [])
                )
                phase = ", ".join(design_mod.get("phases", []))
                start_date = status_mod.get("startDateStruct", {}).get("date", "")
                completion_date = status_mod.get("completionDateStruct", {}).get("date", "")

                results.append({
                    "nct": nct,
                    "title": title,
                    "status": status,
                    "summary": summary,
                    "condition": conditions,
                    "intervention": interventions,
                    "phase": phase,
                    "start_date": start_date,
                    "completion_date": completion_date,
                    "source": "ClinicalTrials.gov",
                    "url": f"https://clinicaltrials.gov/study/{nct}",
                })

            if not results:
                return "No clinical trials found on ClinicalTrials.gov for this query."

            lines = [f"Found {len(results)} clinical trial(s):\n"]
            for t in results:
                lines.append(
                    f"- [{t['title']}](https://clinicaltrials.gov/study/{t['nct']}) "
                    f"| NCT: {t['nct']} | Status: {t['status']} | Phase: {t['phase']}\n"
                    f"  Conditions: {t['condition']}\n"
                    f"  Interventions: {t['intervention']}\n"
                    f"  Summary: {t['summary'][:300]}{'...' if len(t['summary']) > 300 else ''}"
                )
            return "\n\n".join(lines)

        except Exception as e:
            return f"ClinicalTrials.gov error: {e}"
