# med_assistant/tools/pubmed.py

import os
import requests
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any
from med_agent.config.settings import PUBMED_RETMAX, NCBI_API_KEY
from med_agent.tools.base import MedicalTool

# NCBI E‑utilities endpoints
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{BASE_URL}/esearch.fcgi"
EFETCH_URL  = f"{BASE_URL}/efetch.fcgi"

class PubMedSearchTool(MedicalTool):    
    
    """Tool for searching PubMed medical articles with comprehensive results.
    
    This tool performs an advanced search on PubMed to find relevant medical literature
    using MeSH terms and optimized query construction. It handles various medical topics including:
    - Drug effects and safety
    - Medical conditions and symptoms
    - Treatment outcomes
    - Clinical studies and trials
    - Medical procedures
    """    
    name: str = "PubMed Medical Search"
    description: str = "Search PubMed for comprehensive medical information and research articles."

    def build_query(self, query: str) -> str:
        """Build an optimized PubMed search query using MeSH terms and field-specific searching.
        
        Args:
            query (str): The medical query to search for
            
        Returns:
            str: An optimized PubMed search query using MeSH terms where applicable
        """
        terms = query.lower().split()
        field_queries = []
          # Map common terms to MeSH terms for comprehensive medical search
        mesh_mapping = {
            # Drug-related terms
            "side": "adverse effects[MeSH Subheading]",
            "effects": "adverse effects[MeSH Subheading]",
            "adverse": "adverse effects[MeSH Subheading]",
            "safety": "drug safety[MeSH Terms]",
            "dosage": "administration and dosage[MeSH Subheading]",
            "risk": "risk[MeSH Terms]",
            
            # Common medications
            "ibuprofen": "ibuprofen[MeSH Terms]",
            "aspirin": "aspirin[MeSH Terms]",
            "paracetamol": "acetaminophen[MeSH Terms]",
            
            # Medical conditions and symptoms
            "pain": "pain[MeSH Terms]",
            "fever": "fever[MeSH Terms]",
            "inflammation": "inflammation[MeSH Terms]",
            "symptoms": "signs and symptoms[MeSH Terms]",
            
            # Treatment-related terms
            "treatment": "therapeutics[MeSH Terms]",
            "therapy": "therapy[MeSH Subheading]",
            "prevention": "prevention and control[MeSH Subheading]",
            
            # Study types
            "common": "epidemiology[MeSH Subheading]",
            "trial": "clinical trial[Publication Type]",
            "review": "review[Publication Type]",
            "meta": "meta-analysis[Publication Type]"
        }
        
        # Build field-specific queries
        for term in terms:
            if term in mesh_mapping:
                field_queries.append(mesh_mapping[term])
            else:
                field_queries.append(f"{term}[All Fields]")
        
        # Combine with AND
        return " AND ".join(field_queries)

    def handle_rate_limit(self, delay: float = 6.0):
        """Handle rate limit by implementing exponential backoff."""
        import time
        time.sleep(delay)  # Wait for the specified delay

    def _run(self, **kwargs) -> Dict[str, Any]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query (str): The search query for PubMed
            
        Returns:
            dict: Dictionary containing PMIDs or error message
        """
        query = kwargs.get("query", "")
        if not query or not isinstance(query, str):
            return {"pmids": ""}

        max_retries = 3
        base_delay = 6.0  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Set up base parameters
                params = {
                    "db": "pubmed",
                    "retmax": PUBMED_RETMAX,
                    "retmode": "json",
                    "usehistory": "y",
                    "tool": "med_agent",
                    "email": "helloahmedkhawaja@gmail.com"
                }
                
                if NCBI_API_KEY:
                    params["api_key"] = NCBI_API_KEY
                
                # Try the optimized query
                params["term"] = self.build_query(query)
                resp = requests.get(ESEARCH_URL, params=params, timeout=10)
                resp.raise_for_status()
                
                try:
                    data = resp.json()
                    if "esearchresult" in data and "idlist" in data["esearchresult"]:
                        id_list = data["esearchresult"]["idlist"]
                    else:
                        id_list = []
                except json.JSONDecodeError:
                    # Fallback to XML parsing if JSON fails
                    root = ET.fromstring(resp.text)
                    id_list = [id_elem.text for id_elem in root.findall(".//Id")]
                
                # If no results, try with a simpler query
                if not id_list:
                    simple_query = " AND ".join([term for term in query.split() if term.lower() not in ["the", "a", "an"]])
                    params["term"] = simple_query
                    resp = requests.get(ESEARCH_URL, params=params, timeout=10)
                    resp.raise_for_status()
                    
                    try:
                        data = resp.json()
                        if "esearchresult" in data and "idlist" in data["esearchresult"]:
                            id_list = data["esearchresult"]["idlist"]
                    except json.JSONDecodeError:
                        root = ET.fromstring(resp.text)
                        id_list = [id_elem.text for id_elem in root.findall(".//Id")]
                
                print(f"Found {len(id_list)} results for query: {query}")
                return {"pmids": ",".join(id_list[:PUBMED_RETMAX])}
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    self.handle_rate_limit(delay)
                    continue
                print(f"PubMed search error: {str(e)}")
                return {"pmids": "", "error": f"PubMed search error: {str(e)}"}

            except Exception as e:
                print(f"PubMed search error: {str(e)}")
                return {"pmids": "", "error": f"PubMed search error: {str(e)}"}

class PubMedFetchTool(MedicalTool):
    """Tool for fetching and analyzing PubMed medical abstracts.
    
    This tool retrieves detailed medical information from PubMed abstracts including:
    - Main findings and conclusions
    - Study methods and population
    - Clinical significance
    - Treatment recommendations
    - Safety information
    """    
    name: str = "PubMed Medical Information Retrieval"
    description: str = "Fetch and analyze medical information from PubMed articles."

    def _run(self, pmids: str) -> Dict[str, Any]:
        """
        Fetch and process medical information from PubMed abstracts.
        
        This method retrieves abstracts and extracts key medical information including:
        - Clinical findings and outcomes
        - Treatment effectiveness
        - Safety data and adverse effects
        - Patient populations studied
        - Level of evidence
        
        Args:
            pmids (str): Comma-separated list of PubMed IDs
            
        Returns:
            dict: Dictionary containing structured medical information from the abstracts
        """
        if not pmids.strip():
            return {"abstracts": ""}

        try:
            params = {
                "db": "pubmed",
                "id": pmids,
                "retmode": "xml",
                "tool": "med_agent",
                "email": "helloahmedkhawaja@gmail.com"
            }
            
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY
                
            resp = requests.get(EFETCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            
            # Process each article
            articles = []
            for article in root.findall(".//PubmedArticle"):
                formatted_article = self._format_article(article)
                articles.append(formatted_article)
            
            if not articles:
                return {"abstracts": "", "error": "No articles found"}
            
            # Format articles for display
            formatted_texts = [self._format_for_display(article) for article in articles]
            
            # Join with clear separators
            separator = "\n\n" + "="*50 + "\n\n"
            return {
                "abstracts": separator.join(formatted_texts),
                "article_count": len(articles)
            }
        
        except Exception as e:
            return {"abstracts": "", "error": f"Abstract fetch error: {str(e)}"}
    
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
        """Format an article dictionary into a readable string.
        
        Args:
            article: Dictionary containing article information
            
        Returns:
            str: Formatted article text
        """
        sections = []
        
        # Basic information
        sections.append(f"PMID: {article['pmid']}")
        sections.append(f"Title: {article['title']}")
        sections.append(f"Journal: {article['journal']}")
        
        # Authors
        if article['authors']:
            sections.append(f"Authors: {'; '.join(article['authors'][:3])}")
            if len(article['authors']) > 3:
                sections.append(f"    and {len(article['authors']) - 3} more")
        
        # Date
        date_parts = []
        for part in ['year', 'month', 'day']:
            if part in article['date']:
                date_parts.append(article['date'][part])
        if date_parts:
            sections.append(f"Date: {' '.join(date_parts)}")
        
        # DOI
        if article['doi']:
            sections.append(f"DOI: {article['doi']}")
        
        # Publication Types
        if article['publication_types']:
            sections.append(f"Publication Type: {', '.join(article['publication_types'])}")
        
        # MeSH Terms
        if article['mesh_terms']:
            sections.append("MeSH Terms:")
            sections.append("    " + "; ".join(article['mesh_terms']))
        
        # Abstract
        sections.append("\nAbstract:")
        for label, text in article['abstract_sections'].items():
            sections.append(f"\n{label.title()}:")
            sections.append(text)
        
        return "\n".join(sections)
