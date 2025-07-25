from med_agent.tools.base import MedicalTool
from typing import List, Dict
import requests
import logging
import urllib.parse

class ClinicalTrialsSearch(MedicalTool):
    """
    Fetches and summarizes relevant clinical trial evidence from ClinicalTrials.gov for a given query.
    """
    name: str = "ClinicalTrialsSearch"
    description: str = "Search ClinicalTrials.gov for relevant trials"

    def _run(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Args:
            query (str): The user's medical question or search terms.
            max_results (int): Maximum number of trials to fetch.
        Returns:
            List[Dict]: List of trial summaries with title, status, summary, and NCT number.
        """
        logging.debug(f"Searching ClinicalTrials.gov for: {query}")
        base_url = "https://clinicaltrials.gov/api/query/study_fields"
        encoded_query = urllib.parse.quote(query)
        params = {
            "expr": query,
            "fields": "NCTId,BriefTitle,OverallStatus,BriefSummary",
            "min_rnk": 1,
            "max_rnk": max_results,
            "fmt": "json"
        }
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
            results = []
            for study in studies:
                nct = study.get("NCTId", [None])[0]
                title = study.get("BriefTitle", [None])[0]
                status = study.get("OverallStatus", [None])[0]
                summary = study.get("BriefSummary", [None])[0]
                condition = ", ".join(study.get("Condition", []))
                phase = ", ".join(study.get("Phase", []))
                study_type = ", ".join(study.get("StudyType", []))
                start_date = ", ".join(study.get("StartDate", []))
                completion_date = ", ".join(study.get("CompletionDate", []))
                location_city = ", ".join(study.get("LocationCity", []))
                location_country = ", ".join(study.get("LocationCountry", []))
                results.append({
                    "nct": nct,
                    "title": title,
                    "status": status,
                    "summary": summary,
                    "condition": condition,
                    "phase": phase,
                    "study_type": study_type,
                    "start_date": start_date,
                    "completion_date": completion_date,
                    "location_city": location_city,
                    "location_country": location_country,
                    "source": "ClinicalTrials.gov"
                })
            if not results:
                logging.info(f"No clinical trials found for query: {query}")
                return [{
                    "title": "No clinical trials found for this query.",
                    "status": "N/A",
                    "summary": "No results were returned from ClinicalTrials.gov. Please try a different or broader query.",
                    "nct": "N/A",
                    "source": "ClinicalTrials.gov"
                }]
            return results
        except requests.HTTPError as e:
            logging.error(f"ClinicalTrials.gov fetch error: {e} | URL: {resp.url}")
            return [{
                "title": "ClinicalTrials.gov API error.",
                "status": "N/A",
                "summary": f"Error: {e}. Please check the query or try again later.",
                "nct": "N/A",
                "source": "ClinicalTrials.gov"
            }]
        except Exception as e:
            logging.error(f"ClinicalTrials.gov fetch error: {e}")
            return [{
                "title": "ClinicalTrials.gov API error.",
                "status": "N/A",
                "summary": f"Error: {e}. Please check the query or try again later.",
                "nct": "N/A",
                "source": "ClinicalTrials.gov"
            }]
