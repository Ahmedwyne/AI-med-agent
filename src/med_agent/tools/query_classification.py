def classify_query_type(query: str) -> str:
    """
    Classify the user's medical question type: diagnosis, treatment, prognosis, drug info, etc.
    Returns a string label for the query type.
    """
    query_lower = query.lower()
    if any(word in query_lower for word in ["diagnose", "diagnosis", "differential", "identify"]):
        return "diagnosis"
    if any(word in query_lower for word in ["treat", "treatment", "manage", "therapy", "intervention"]):
        return "treatment"
    if any(word in query_lower for word in ["prognosis", "outcome", "survival", "risk of recurrence"]):
        return "prognosis"
    if any(word in query_lower for word in ["drug", "medication", "dose", "side effect", "adverse", "pharmacology"]):
        return "drug_info"
    if any(word in query_lower for word in ["prevention", "prevent", "screening"]):
        return "prevention"
    return "general"
