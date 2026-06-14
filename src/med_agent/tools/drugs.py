import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from med_agent.tools.base import MedicalTool, http_session
from med_agent.tools.cache import get_cached

logger = logging.getLogger(__name__)

_RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"

class DrugInfoTool(MedicalTool):
    """Tool for retrieving drug information from RxNorm."""
    name: str = "Drug Information"
    description: str = "Get drug information from RxNorm."

    def _run(self, query: str) -> str:
        if not query:
            return "No drug name provided."
        return get_cached(f"DrugInfo:{query.lower()}", lambda: self._fetch(query), ttl=86400)

    def _fetch(self, query: str) -> str:
        try:
            # Resolve RxCUI
            search = http_session.get(f"{_RXNAV_BASE}/rxcui.json", params={"name": query}, timeout=5)
            search.raise_for_status()
            rxcui_list = search.json().get("idGroup", {}).get("rxnormId", [])
            if not rxcui_list:
                return f"No RxCUI found for '{query}' in RxNorm."
            rxcui = rxcui_list[0]
            logger.debug(f"RxNorm: resolved '{query}' to RxCUI {rxcui}")

            # Get drug properties
            props = http_session.get(f"{_RXNAV_BASE}/rxcui/{rxcui}/properties.json", timeout=5)
            props.raise_for_status()
            prop_data = props.json().get("properties", {})
            name = prop_data.get("name")
            tty = prop_data.get("tty")

            synonyms: list[str] = []
            brands: list[str] = []

            # Fetch related terms only for non-ingredient entries; run three calls in parallel
            if tty and tty != "IN":
                def _fetch_related(rel_type: str):
                    r = http_session.get(
                        f"{_RXNAV_BASE}/rxcui/{rxcui}/related.json",
                        params={"tty": rel_type},
                        timeout=5,
                    )
                    r.raise_for_status()
                    return rel_type, r.json().get("relatedGroup", {}).get("conceptGroup", [])

                with ThreadPoolExecutor(max_workers=3) as ex:
                    futures = {ex.submit(_fetch_related, t): t for t in ["SY", "BN", "BPCK"]}
                    for future in as_completed(futures):
                        rel_type, concept_groups = future.result()
                        for group in concept_groups:
                            group_tty = group.get("tty")
                            for concept in group.get("conceptProperties", []):
                                term = concept.get("name")
                                if group_tty == "SY" and term and term not in synonyms:
                                    synonyms.append(term)
                                elif group_tty in ["BN", "BPCK"] and term and term not in brands:
                                    brands.append(term)
            else:
                logger.debug(f"Skipping related terms for ingredient-level RxCUI (TTY=IN): {rxcui}")

            reasoning = []
            if tty == "IN":
                reasoning.append(f"{name} is an ingredient-level entry in RxNorm, representing the active substance.")
            elif tty:
                reasoning.append(f"{name} is classified as '{tty}' in RxNorm, which may indicate a brand, pack, or synonym.")
            if brands:
                reasoning.append(f"Common brand names include: {', '.join(brands)}.")
            if synonyms:
                reasoning.append(f"Synonyms or alternative names: {', '.join(synonyms[:5]) + ('...' if len(synonyms) > 5 else '')}.")
            if not brands and not synonyms:
                reasoning.append("No brand or synonym information was found for this entry.")
            reasoning.append("Always verify drug information with a healthcare provider or pharmacist, especially for dosing, interactions, and contraindications.")

            plain_summary = f"{name} (RxCUI: {rxcui}) is a {tty or 'drug'} used in clinical practice. "
            if brands:
                plain_summary += f"It is available under brand names such as {', '.join(brands[:3]) + ('...' if len(brands) > 3 else '')}. "
            if synonyms:
                plain_summary += f"It may also be known as {', '.join(synonyms[:3]) + ('...' if len(synonyms) > 3 else '')}. "
            plain_summary += "Consult a healthcare professional for detailed usage, safety, and interaction information."

            return (
                f"Drug Name: {name}\n"
                f"Term Type: {tty or 'N/A'}\n"
                f"RxCUI: {rxcui}\n"
                f"Synonyms: {', '.join(synonyms) if synonyms else 'N/A'}\n"
                f"Brand Names: {', '.join(brands) if brands else 'N/A'}\n"
                f"\nClinical Reasoning & Relevance:\n- " + "\n- ".join(reasoning) +
                f"\n\nPlain-language summary:\n{plain_summary}"
            )

        except Exception as e:
            logger.error(f"Drug lookup error for '{query}': {e}")
            return f"Drug lookup error: {str(e)}"
