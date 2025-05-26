import sys
from pathlib import Path
import unittest

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from med_agent.crew import research_agent

class TestMedAgentResearch(unittest.TestCase):
    def test_pubmed_search(self):
        from crewai import Task
        
        query = "common side effects of ibuprofen"
        task = Task(
            description=f"Research and provide a comprehensive answer about: {query}. Include relevant PubMed article IDs (PMIDs) as references for each major point.",
            agent=research_agent,
            expected_output="A detailed summary with PMIDs as references"
        )
        result = research_agent.execute_task(task)
        print(f"\nQuery: {query}\nAnswer: {result}")
        
        # Basic validation
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0, "Result should not be empty.")
        
        # Check for references
        has_references = any(ref in result for ref in ["PMID", "Reference", "references"])
        self.assertTrue(has_references, "Answer should include references")
        
        # Check for key content
        key_terms = ["side effect", "adverse", "ibuprofen"]
        has_content = any(term.lower() in result.lower() for term in key_terms)
        self.assertTrue(has_content, "Answer should discuss side effects")

if __name__ == "__main__":
    unittest.main()
