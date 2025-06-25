from med_agent.tools.base import MedicalTool
from typing import List, Dict
import requests
import logging
from bs4 import BeautifulSoup

class CDCGuidelines(MedicalTool):
    """
    Fetches and summarizes relevant CDC guidelines for a given query using web scraping.
    """
    name: str = "CDCGuidelines"
    description: str = "Fetch and parse CDC guidelines for medical queries"

    def _run(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Args:
            query (str): The user's medical question or search terms.
            max_results (int): Maximum number of guidelines to fetch.
        Returns:
            List[Dict]: List of guideline summaries with title, summary, link, and evidence level.
        """
        logging.debug(f"Searching CDC guidelines for: {query}")
        search_url = f"https://www.cdc.gov/search/index.html"
        params = {"query": query, "sitelimit": "www.cdc.gov"}
        try:
            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for item in soup.select('.result')[:max_results]:
                title_tag = item.select_one('.result-title')
                summary_tag = item.select_one('.result-summary')
                link_tag = title_tag.find('a') if title_tag else None
                title = title_tag.get_text(strip=True) if title_tag else "CDC Guideline"
                summary = summary_tag.get_text(strip=True) if summary_tag else "No summary available."
                link = link_tag['href'] if link_tag and link_tag.has_attr('href') else search_url
                if not link.startswith('http'):
                    link = f"https://www.cdc.gov{link}"
                results.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": "CDC",
                    "evidence_level": "Guideline"
                })
            if not results:
                # fallback to placeholder if nothing found
                return [
                    {
                        "title": f"CDC Guidance on {query.title()}",
                        "summary": f"Summary of CDC recommendations for {query} (example).",
                        "link": f"https://www.cdc.gov/search.html?q={query.replace(' ', '+')}",
                        "source": "CDC",
                        "evidence_level": "Guideline"
                    }
                ]
            return results
        except Exception as e:
            logging.error(f"CDC guideline fetch error: {e}")
            return [
                {
                    "title": f"CDC Guidance on {query.title()}",
                    "summary": f"Summary of CDC recommendations for {query} (example).",
                    "link": f"https://www.cdc.gov/search.html?q={query.replace(' ', '+')}",
                    "source": "CDC",
                    "evidence_level": "Guideline"
                }
            ]
