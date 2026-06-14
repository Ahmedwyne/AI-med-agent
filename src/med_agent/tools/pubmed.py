import time
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import logging
import requests
from med_agent.tools.base import MedicalTool, http_session
from med_agent.tools.cache import get_cached
from med_agent.config.settings import NCBI_API_KEY, NCBI_EMAIL

logger = logging.getLogger(__name__)

if not NCBI_EMAIL:
    logging.warning("NCBI_EMAIL is not set. Set NCBI_EMAIL in .env to comply with NCBI ToS.")

# NCBI E‑utilities endpoints
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{BASE_URL}/esearch.fcgi"
EFETCH_URL = f"{BASE_URL}/efetch.fcgi"

PUBMED_RETMAX = 5  # Keep small: fewer articles = less context, faster pipeline

class PubMedSearch(MedicalTool):
    """Search PubMed for medical articles."""
    name: str = "PubMedSearch"
    description: str = "Search PubMed for medical articles."

    def build_query(self, query: str) -> str:
        # Pass the query as-is; PubMed handles natural language and phrase search well.
        # Wrapping every word in [All Fields] AND ... is overly restrictive.
        return query

    def handle_rate_limit(self, delay: float = 6.0):
        time.sleep(delay)

    def _run(self, query: str = "", **kwargs) -> Dict[str, Any]:
        if not query:
            query = kwargs.get("query", "")
        if not query:
            logger.debug("PubMedSearch._run: No valid query provided.")
            return {"pmids": [], "count": 0, "query": query}
        return get_cached(f"PubMedSearch:{query}", lambda: self._fetch(query), ttl=3600)

    def _fetch(self, query: str) -> Dict[str, Any]:
        max_retries = 3
        base_delay = 6.0
        for attempt in range(max_retries):
            try:
                params = {
                    "db": "pubmed",
                    "retmax": PUBMED_RETMAX,
                    "retmode": "json",
                    "usehistory": "y",
                    "tool": "med_agent",
                    "email": NCBI_EMAIL,
                }
                if NCBI_API_KEY:
                    params["api_key"] = NCBI_API_KEY
                params["term"] = self.build_query(query)
                resp = http_session.get(ESEARCH_URL, params=params, timeout=10)
                resp.raise_for_status()
                try:
                    data = resp.json()
                    if "esearchresult" in data and "idlist" in data["esearchresult"]:
                        id_list = data["esearchresult"]["idlist"]
                    else:
                        id_list = []
                except json.JSONDecodeError:
                    root = ET.fromstring(resp.text)
                    id_list = [id_elem.text for id_elem in root.findall(".//Id")]
                pmids = id_list[:PUBMED_RETMAX]
                logger.info(f"PubMed search: {len(pmids)} results for query: {query!r}")
                if not pmids:
                    return {"pmids": [], "count": 0, "query": query, "message": "No PubMed articles found for this query."}
                return {"pmids": pmids, "count": len(pmids), "query": query}
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.handle_rate_limit(delay)
                    continue
                return {"pmids": [], "count": 0, "query": query, "error": str(e)}
            except Exception as e:
                return {"pmids": [], "count": 0, "query": query, "error": str(e)}

class PubMedFetch(MedicalTool):
    """Fetch and analyze medical information from PubMed articles."""
    name: str = "PubMedFetch"
    description: str = "Fetch and analyze medical information from PubMed articles."

    def _run(self, pmids: list = None, **kwargs) -> str:
        if pmids is None:
            pmids = kwargs.get("pmids", [])
        if isinstance(pmids, str):
            pmids = [p.strip() for p in pmids.split(",") if p.strip()]
        if not pmids:
            return "No PMIDs provided to fetch."
        cache_key = f"PubMedFetch:{','.join(sorted(pmids))}"
        return get_cached(cache_key, lambda: self._fetch(pmids), ttl=86400)

    def _fetch(self, pmids: list) -> str:
        try:
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "tool": "med_agent",
                "email": NCBI_EMAIL,
            }
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY
            resp = http_session.get(EFETCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            articles = []
            for article in root.findall(".//PubmedArticle"):
                formatted_article = self._format_article(article)
                articles.append(formatted_article)
            if not articles:
                return "No articles found for the provided PMIDs."
            separator = "\n\n" + "=" * 50 + "\n\n"
            return separator.join(self._format_for_display(a) for a in articles)
        except Exception as e:
            return f"Abstract fetch error: {str(e)}"
    
    def _format_article(self, article: ET.Element) -> Dict[str, Any]:
        """Format a PubMed article into a structured dictionary.
        
        Args:
            article: XML element containing the PubMed article
            
        Returns:
            dict: Structured article information
        """
        # Basic article information
        result = {
            "pmid": article.find(".//PMID").text if article.find(".//PMID") is not None else "N/A",
            "title": article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "N/A",
            "journal": article.find(".//Journal/Title").text if article.find(".//Journal/Title") is not None else "N/A",
            "publication_types": [],
            "mesh_terms": [],
            "abstract_sections": {},
            "authors": [],
            "date": {},
            "doi": None
        }
        
        # Get authors
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            firstname = author.find("ForeName")
            if lastname is not None and firstname is not None:
                result["authors"].append(f"{lastname.text}, {firstname.text}")
        
        # Get publication types
        for pub_type in article.findall(".//PublicationType"):
            if pub_type.text:
                result["publication_types"].append(pub_type.text)
        
        # Get MeSH terms
        for mesh in article.findall(".//MeshHeading"):
            descriptor = mesh.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                result["mesh_terms"].append(descriptor.text)
        
        # Get publication date
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            for date_part in ["Year", "Month", "Day"]:
                element = pub_date.find(f".//{date_part}")
                if element is not None:
                    result["date"][date_part.lower()] = element.text
        
        # Get DOI
        article_ids = article.findall(".//ArticleId")
        for article_id in article_ids:
            if article_id.get("IdType") == "doi":
                result["doi"] = article_id.text
                break
        
        # Get abstract sections
        abstract = article.find(".//Abstract")
        if abstract is not None:
            for section in abstract.findall(".//AbstractText"):
                label = section.get("Label", "Background").lower()
                text = section.text or ""
                result["abstract_sections"][label] = text
            
            # If no structured sections, get plain abstract
            if not result["abstract_sections"]:
                abstract_text = " ".join([text.text or "" for text in abstract.findall(".//AbstractText")])
                if abstract_text:
                    result["abstract_sections"]["text"] = abstract_text
        
        return result

    def _format_for_display(self, article: Dict[str, Any]) -> str:
        """Concise format — keeps token usage low for the LLM context."""
        year = article['date'].get('year', '')
        pub_type = article['publication_types'][0] if article['publication_types'] else ''
        header = f"[PMID:{article['pmid']}] {article['title']} ({pub_type}, {year})"

        # Pick the most informative abstract section (≤400 chars)
        sections = article['abstract_sections']
        text = (
            sections.get('conclusion')
            or sections.get('results')
            or sections.get('text')
            or next(iter(sections.values()), '')
        )
        text = text[:400] + ('...' if len(text) > 400 else '')

        return f"{header}\nAbstract: {text}"
