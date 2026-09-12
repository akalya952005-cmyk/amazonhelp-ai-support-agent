"""
Historical Conversation Retriever using Sentence Embeddings & FAISS.
Retrieves top-5 most similar AmazonHelp customer interactions to serve as RAG context.
"""

import os
import logging
import importlib
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _get_numpy():
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


def _get_faiss():
    try:
        return importlib.import_module("faiss")
    except Exception:
        return None


class HistoricalRetriever:
    """
    Dense semantic retriever powered by Sentence-Transformers and FAISS IndexFlatIP.
    Gracefully falls back to TF-IDF Cosine Similarity if dense packages are unavailable.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "data"):
        self.model_name = model_name
        self.index_dir = index_dir
        self.corpus_df: Optional[pd.DataFrame] = None
        self.faiss_index = None
        self.embedder = None
        self.use_dense = True
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self._init_embedder()

    def _init_embedder(self):
        try:
            st = importlib.import_module("sentence_transformers")
            faiss = _get_faiss()
            if faiss is None:
                raise ImportError("faiss not available")
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.embedder = st.SentenceTransformer(self.model_name)
            self.use_dense = True
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformers or FAISS ({e}). Falling back to TF-IDF semantic retriever.")
            self.use_dense = False

    def build_index(self, df: pd.DataFrame, force_rebuild: bool = False):
        """Builds index over historical customer queries and stores reference replies."""
        os.makedirs(self.index_dir, exist_ok=True)
        index_file = os.path.join(self.index_dir, "faiss_index.bin")
        corpus_file = os.path.join(self.index_dir, "corpus_reference.parquet")

        self.corpus_df = df.copy().reset_index(drop=True)

        if not force_rebuild and os.path.exists(index_file) and os.path.exists(corpus_file) and self.use_dense:
            try:
                faiss = _get_faiss()
                if faiss is not None:
                    logger.info(f"Loading FAISS index from {index_file}...")
                    self.faiss_index = faiss.read_index(index_file)
                    self.corpus_df = pd.read_parquet(corpus_file)
                    logger.info(f"Retriever initialized with {len(self.corpus_df)} records from cache.")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached FAISS index: {e}. Rebuilding index...")

        if self.use_dense and self.embedder is not None:
            try:
                faiss = _get_faiss()
                np = _get_numpy()
                if faiss is None or np is None:
                    raise ImportError("faiss or numpy not available")
                logger.info(f"Computing dense embeddings for {len(self.corpus_df)} historical tweets...")
                queries = self.corpus_df["clean_text"].fillna("").tolist()
                embeddings = self.embedder.encode(queries, show_progress_bar=True, normalize_embeddings=True)
                embeddings = np.ascontiguousarray(embeddings.astype("float32"))

                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(embeddings)

                faiss.write_index(self.faiss_index, index_file)
                self.corpus_df.to_parquet(corpus_file, index=False)
                logger.info(f"Successfully built and cached FAISS index at {index_file}")
                return
            except Exception as e:
                logger.warning(f"Dense index building failed: {e}. Falling back to TF-IDF cosine retriever.")
                self.use_dense = False

        # Fallback: TF-IDF Cosine Similarity
        try:
            sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
            TfidfVectorizer = sklearn_text.TfidfVectorizer
            logger.info(f"Building TF-IDF retriever index for {len(self.corpus_df)} records...")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_df["clean_text"].fillna(""))
            logger.info("TF-IDF retriever index ready.")
        except Exception as e:
            logger.error(f"Failed to initialize fallback TF-IDF retriever: {e}")

    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar historical customer interactions.
        Returns list of dicts with customer message, agent reply, intent, and similarity score.
        """
        if self.corpus_df is None or len(self.corpus_df) == 0:
            raise RuntimeError("Index is empty. Call build_index() first.")

        results = []

        if self.use_dense and self.faiss_index is not None and self.embedder is not None:
            np = _get_numpy()
            if np is not None:
                query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
                query_emb = np.ascontiguousarray(query_emb)
                scores, indices = self.faiss_index.search(query_emb, top_k)

                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.corpus_df):
                        continue
                    row = self.corpus_df.iloc[idx]
                    results.append({
                        "customer_message": str(row.get("clean_text", row.get("text", ""))),
                        "agent_reply": str(row.get("human_reply", row.get("clean_reply", ""))),
                        "intent": str(row.get("intent", "Other")),
                        "similarity_score": round(float(score), 4)
                    })
                return results

        # TF-IDF fallback retrieval
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                pairwise = importlib.import_module("sklearn.metrics.pairwise")
                cosine_similarity = pairwise.cosine_similarity
                query_vec = self.tfidf_vectorizer.transform([query])
                sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
                # Rank scores in pure Python without requiring static numpy import
                top_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:top_k]

                for idx in top_indices:
                    row = self.corpus_df.iloc[idx]
                    results.append({
                        "customer_message": str(row.get("clean_text", row.get("text", ""))),
                        "agent_reply": str(row.get("human_reply", row.get("clean_reply", ""))),
                        "intent": str(row.get("intent", "Other")),
                        "similarity_score": round(float(sim_scores[idx]), 4)
                    })
                return results
            except Exception as e:
                logger.error(f"TF-IDF similarity error: {e}")

        return []

    def format_examples_for_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Formats retrieved examples cleanly for Claude API prompt template."""
        formatted_blocks = []
        for i, ex in enumerate(examples, 1):
            block = (
                f"Example {i} [Intent: {ex['intent']}]:\n"
                f"Customer: {ex['customer_message']}\n"
                f"AmazonHelp: {ex['agent_reply']}"
            )
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)
