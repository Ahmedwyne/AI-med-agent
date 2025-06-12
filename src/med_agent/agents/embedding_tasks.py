import os
import pickle
import re
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from med_agent.config.settings import EMBED_MODEL, VECTOR_DIR, GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS
from med_agent.tools.base import MedicalTool

# Paths for persistence
INDEX_PATH   = os.path.join(VECTOR_DIR, "faiss_index.bin")
CHUNKS_PATH  = os.path.join(VECTOR_DIR, "chunks.pkl")

# Ensure vector directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)

# Initialize models
embedder   = SentenceTransformer(EMBED_MODEL)
groq_client = Groq(api_key=GROQ_API_KEY)


def sentence_chunk(text: str, max_sentences: int = 3, overlap: int = 1) -> List[str]:
    """Split text into overlapping sentence-based chunks."""
    sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s]
    chunks = []
    step = max_sentences - overlap if max_sentences > overlap else 1
    for i in range(0, len(sents), step):
        chunks.append(" ".join(sents[i : i + max_sentences]))
    return chunks


class EmbedAndIndexTool(MedicalTool):
    """Tool for embedding and indexing text."""    
    name: str = "Embed and Index"
    description: str = "Embed text chunks and store them in a vector database."

    def _run(self, text: str) -> Dict[str, Any]:
        """
        Embeds text chunks and builds (or updates) a FAISS index,
        persisting both the index and the chunks-to-ID mapping.
        
        Args:
            text (str): Text to embed and index
            
        Returns:
            dict: Processing status and number of chunks indexed
        """
        if not text.strip():
            return {"status": "no_text"}

        chunks = sentence_chunk(text)
        embeddings = embedder.encode(chunks).astype("float32")

        # Build a FlatL2 index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Persist index and chunks
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

        return {"status": "indexed", "num_chunks": len(chunks)}


class RetrieveChunksTool(MedicalTool):
    """Tool for retrieving relevant text chunks."""    
    name: str = "Retrieve Chunks"
    description: str = "Find relevant chunks of text for a query."

    def _run(self, query: str) -> Dict[str, Any]:
        """
        Loads the FAISS index and chunk list, then
        performs a k-NN search to find the top 3 chunks.
        
        Args:
            query (str): Query to search for
            
        Returns:
            dict: Dictionary containing relevant text chunks
        """
        if not query.strip() or not os.path.exists(INDEX_PATH):
            return {"contexts": []}

        # Load index and chunks
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        # Embed & search
        q_embed = embedder.encode([query]).astype("float32")
        distances, ids = index.search(q_embed, k=3)
        top_chunks = [chunks[i] for i in ids[0] if i < len(chunks)]

        return {"contexts": top_chunks}


class GenerateSummaryTool(MedicalTool):
    """Tool for generating medical summaries."""    
    name: str = "Generate Summary"
    description: str = "Generate a medical summary from context and query."

    def _run(self, query: str, contexts: List[str] = None) -> Dict[str, Any]:
        """
        Uses Groq-hosted LLaMA to synthesize the final answer
        from retrieved contexts and the original question.
        If GROQ_API_KEY is missing or MOCK_LLM=1 is set, returns a mock answer for testing.
        """
        # Filter out empty/whitespace-only chunks
        if not contexts or not query.strip():
            return {"answer": "No relevant medical evidence was found for your query. Please try rephrasing or consult a healthcare professional."}
        
        # Limit to top 2-3 non-empty, truncated chunks (max 500 chars each)
        filtered = [c.strip()[:500] for c in contexts if c and c.strip()]
        if not filtered:
            return {"answer": "No relevant medical evidence was found for your query. Please try rephrasing or consult a healthcare professional."}
        top_contexts = filtered[:3]
        context_str = "\n\n".join(top_contexts)

        # Mock LLM for local testing if needed
        use_mock = os.getenv("MOCK_LLM", "0") == "1" or not GROQ_API_KEY
        if use_mock:
            mock_answer = (
                "[MOCK LLM] Evidence-based summary for: '" + query + "'\n"
                "Key findings from context:\n" + '\n'.join(f"- {c[:80]}..." for c in top_contexts) +
                "\n(Citations: PMID123456, PMID234567)"
            )
            return {"answer": mock_answer}

        prompt = (
            "You are a medical assistant. Use ONLY the provided context below to answer the user's medical question. "
            "For each major claim, cite the PMID (PubMed ID) or source if available. "
            "If the context does not contain enough information, say: 'No relevant medical evidence was found in PubMed. Please consult a healthcare professional.' "
            "Structure your answer as a concise, evidence-based summary with bullet points or sections. "
            "Do NOT make up information.\n\n"
            f"CONTEXT (from PubMed abstracts, may include PMIDs):\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER (with citations):"
        )
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GROQ_MAX_TOKENS
            )
            answer = resp.choices[0].message.content.strip()
            # Fallback if LLM returns empty or generic answer
            if not answer or answer.lower().startswith("insufficient") or "no relevant" in answer.lower():
                return {"answer": "No relevant medical evidence was found for your query. Please try rephrasing or consult a healthcare professional."}
            return {"answer": answer}
        except Exception as e:
            return {"answer": f"LLM error: {str(e)}"}


if __name__ == "__main__":
    # Embed all .txt files in the knowledge/ folder
    knowledge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../knowledge'))
    all_text = []
    for fname in os.listdir(knowledge_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(knowledge_dir, fname), 'r', encoding='utf-8') as f:
                all_text.append(f.read())
    if all_text:
        print(f"Embedding {len(all_text)} documents from 'knowledge/'...")
        tool = EmbedAndIndexTool()
        result = tool._run("\n".join(all_text))
        print(f"Embedding result: {result}")
    else:
        print("No .txt files found in 'knowledge/' folder. Nothing embedded.")