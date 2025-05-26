import sys
from pathlib import Path
import unittest

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from med_agent.crew import crew

class TestMedAgentCrew(unittest.TestCase):
    def test_ibuprofen_side_effects(self):
        query = "What are the common side effects of ibuprofen?"
        try:
            result = crew.kickoff(inputs={"query": query})
        except Exception as e:
            self.fail(f"Crew run failed with exception: {e}")

        # Try both possible keys for the summary
        summary = (
            result.get("GenerateSummaryTask") or
            result.get("generate_summary") or
            {}
        )
        answer = summary.get("answer", "")
        print(f"Q: {query}\nA: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0, "Answer should not be empty.")

if __name__ == "__main__":
    unittest.main()