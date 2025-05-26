from litellm import completion
import os

queries = [
    "What is the recommended treatment for STEMI?",
    "List the contraindications of metformin.",
    "Summarize recent clinical trials on GLP-1 agonists in diabetes.",
    "What are the latest guidelines for hypertension management?",
    "What is the mechanism of action of aspirin?",
    "What are the side effects of ibuprofen?"
]

for q in queries:
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        messages=[{"role": "user", "content": q}]
    )
    print(f"\nQuery: {q}\nResponse: {response}\n")
