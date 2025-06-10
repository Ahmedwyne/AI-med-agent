import requests
from typing import Dict, Any
from med_agent.config.settings import PUBMED_RETMAX 
from med_agent.tools.base import MedicalTool

class DrugInfoTool(MedicalTool):
    """Tool for retrieving drug information from RxNorm."""    
    name: str = "Drug Information"
    description: str = "Get drug information from RxNorm."

    def _run(self, query: str) -> Dict[str, Any]:
        if not query:
            return {"drug_info": "No drug name provided."}

        try:
            # First, get RxCUI
            print(f"\nSearching for RxCUI for {query}...")
            search = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", 
                                  params={"name": query}, timeout=5)
            search.raise_for_status()
            print(f"Search response: {search.text}")
            rxcui_list = search.json().get("idGroup", {}).get("rxnormId", [])
            if not rxcui_list:
                return {"drug_info": f"No RxCUI found for '{query}'."}
            rxcui = rxcui_list[0]
            print(f"\nFound RxCUI: {rxcui}")

            # Get drug properties
            print(f"Getting drug properties...")
            props = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout=5)
            props.raise_for_status()
            print(f"Properties response: {props.text}")
            prop_data = props.json().get("properties", {})
            name = prop_data.get("name")
            tty = prop_data.get("tty")

            synonyms = []
            brands = []
            # Only fetch related terms if TTY is not 'IN' (ingredient)
            if tty and tty != "IN":
                print(f"Getting related terms...")
                for rel_type in ["SY", "BN", "BPCK"]:
                    rel_resp = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json", 
                                            params={"tty": rel_type}, timeout=5)
                    rel_resp.raise_for_status()
                    rel_data = rel_resp.json().get("relatedGroup", {}).get("conceptGroup", [])
                    for group in rel_data:
                        group_tty = group.get("tty")
                        for concept in group.get("conceptProperties", []):
                            term = concept.get("name")
                            if group_tty == "SY" and term and term not in synonyms:
                                synonyms.append(term)
                            elif group_tty in ["BN", "BPCK"] and term and term not in brands:
                                brands.append(term)
            else:
                print("No related terms for ingredient-level RxCUI (TTY=IN).")

            # Reasoning and clinical relevance
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

            # Plain-language summary
            plain_summary = f"{name} (RxCUI: {rxcui}) is a {tty or 'drug'} used in clinical practice. "
            if brands:
                plain_summary += f"It is available under brand names such as {', '.join(brands[:3]) + ('...' if len(brands) > 3 else '')}. "
            if synonyms:
                plain_summary += f"It may also be known as {', '.join(synonyms[:3]) + ('...' if len(synonyms) > 3 else '')}. "
            plain_summary += "Consult a healthcare professional for detailed usage, safety, and interaction information."

            summary = (
                f"Drug Name: {name}\n"
                f"Term Type: {tty or 'N/A'}\n"
                f"RxCUI: {rxcui}\n"
                f"Synonyms: {', '.join(synonyms) if synonyms else 'N/A'}\n"
                f"Brand Names: {', '.join(brands) if brands else 'N/A'}\n"
                f"\nClinical Reasoning & Relevance:\n- " + "\n- ".join(reasoning) +
                f"\n\nPlain-language summary:\n{plain_summary}"
            )
            return {"drug_info": summary}

        except requests.exceptions.RequestException as e:
            print(f"\nAPI Error: {str(e)}")
            return {"drug_info": f"API Error: {str(e)}"}
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\nError: {str(e)}\n{error_details}")
            return {"drug_info": f"Error: {str(e)}\n{error_details}"}
