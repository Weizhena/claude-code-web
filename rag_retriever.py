"""
Two-Layer RAG Retrieval System

Layer 1: Category-level retrieval
    - Given question q, find the most relevant k categories
    - Uses BM25 or embedding similarity between q and category descriptions/contents

Layer 2: Entry-level retrieval
    - From each selected category, retrieve top n most relevant entries
    - Uses BM25 scoring on entry content

Final output: Retrieved memory entries formatted for QA
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import re
from collections import Counter

from .memory_structure import MemorySystem, MemoryCategory, MemoryEntry


@dataclass
class RetrievalResult:
    """Result of retrieval operation"""
    query: str
    selected_categories: List[Tuple[MemoryCategory, float]]  # (category, score)
    retrieved_entries: List[Tuple[MemoryEntry, float]]  # (entry, score)
    total_retrieved: int


class BM25Scorer:
    """BM25 scoring implementation"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # per-doc term frequencies
        self.N: int = 0  # total number of documents

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[str]) -> None:
        """Fit BM25 on a corpus of documents"""
        self.documents = documents
        self.N = len(documents)
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.doc_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Term frequency for this document
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))

            # Document frequency (unique terms in this doc)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / max(self.N, 1)

    def _idf(self, term: str) -> float:
        """Calculate IDF for a term"""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a document"""
        if doc_idx >= self.N:
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
            score += idf * numerator / denominator

        return score

    def score_all(self, query: str) -> List[Tuple[int, float]]:
        """Score query against all documents, return sorted (doc_idx, score)"""
        scores = []
        for idx in range(self.N):
            score = self.score(query, idx)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class TwoLayerRAGRetriever:
    """
    Two-Layer RAG Retriever

    Layer 1: Select top-k categories based on query relevance
    Layer 2: From each selected category, retrieve top-n entries
    """

    def __init__(self,
                 memory_system: MemorySystem,
                 k_categories: int = 3,
                 n_entries_per_category: int = 5,
                 use_core_memory: bool = True):
        """
        Initialize retriever

        Args:
            memory_system: The memory system to retrieve from
            k_categories: Number of categories to select in Layer 1
            n_entries_per_category: Number of entries per category in Layer 2
            use_core_memory: Whether to always include core memory
        """
        self.memory = memory_system
        self.k_categories = k_categories
        self.n_entries = n_entries_per_category
        self.use_core_memory = use_core_memory

        # BM25 scorers for each layer
        self.category_scorer = BM25Scorer()
        self.entry_scorers: Dict[MemoryCategory, BM25Scorer] = {}

        # Build indices
        self._build_category_index()
        self._build_entry_indices()

    def _build_category_index(self) -> None:
        """Build BM25 index for category-level retrieval"""
        # Create document for each category: description + sample entries
        category_docs = []
        for category in MemoryCategory:
            doc_parts = [
                category.value,
                MemoryCategory.get_description(category),
                MemoryCategory.get_chinese_name(category)
            ]

            # Add sample entries from this category
            entries = self.memory.get_category_entries(category)
            for entry in entries[:10]:  # Sample first 10 entries
                doc_parts.append(entry.content)

            category_docs.append(" ".join(doc_parts))

        self.category_scorer.fit(category_docs)

    def _build_entry_indices(self) -> None:
        """Build BM25 indices for entry-level retrieval per category"""
        for category in MemoryCategory:
            entries = self.memory.get_category_entries(category)
            if entries:
                scorer = BM25Scorer()
                docs = [entry.content for entry in entries]
                scorer.fit(docs)
                self.entry_scorers[category] = scorer

    def refresh_indices(self) -> None:
        """Refresh all indices after memory updates"""
        self._build_category_index()
        self._build_entry_indices()

    def layer1_category_retrieval(self, query: str) -> List[Tuple[MemoryCategory, float]]:
        """
        Layer 1: Retrieve top-k most relevant categories

        Args:
            query: The search query

        Returns:
            List of (category, score) tuples, sorted by relevance
        """
        # Score all categories
        scores = self.category_scorer.score_all(query)

        # Map indices to categories
        categories = list(MemoryCategory)
        results = []
        for idx, score in scores[:self.k_categories]:
            if idx < len(categories):
                results.append((categories[idx], score))

        return results

    def layer2_entry_retrieval(self,
                                query: str,
                                categories: List[MemoryCategory]) -> List[Tuple[MemoryEntry, float]]:
        """
        Layer 2: Retrieve top-n entries from each selected category

        Args:
            query: The search query
            categories: List of categories to search in

        Returns:
            List of (entry, score) tuples, sorted by relevance
        """
        all_results = []

        for category in categories:
            entries = self.memory.get_category_entries(category)
            if not entries:
                continue

            # Get scorer for this category
            scorer = self.entry_scorers.get(category)
            if scorer is None:
                continue

            # Score entries
            scores = scorer.score_all(query)

            # Get top-n entries
            for idx, score in scores[:self.n_entries]:
                if idx < len(entries):
                    all_results.append((entries[idx], score))

        # Sort all results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Two-layer retrieval pipeline

        Args:
            query: The search query

        Returns:
            RetrievalResult containing selected categories and entries
        """
        # Layer 1: Category selection
        selected_categories = self.layer1_category_retrieval(query)

        # Layer 2: Entry retrieval from selected categories
        category_list = [cat for cat, _ in selected_categories]
        retrieved_entries = self.layer2_entry_retrieval(query, category_list)

        return RetrievalResult(
            query=query,
            selected_categories=selected_categories,
            retrieved_entries=retrieved_entries,
            total_retrieved=len(retrieved_entries)
        )

    def retrieve_for_qa(self, query: str) -> str:
        """
        Retrieve and format memories for question answering

        Args:
            query: The question

        Returns:
            Formatted string with retrieved memories
        """
        result = self.retrieve(query)

        sections = []

        # Always include core memory if enabled
        if self.use_core_memory and self.memory.core_memory.content:
            sections.append(f"=== CORE MEMORY ===\n{self.memory.core_memory.content}")

        # Add retrieved entries grouped by category
        category_entries: Dict[str, List[str]] = {}

        for entry, score in result.retrieved_entries:
            cat_name = entry.category.value
            if cat_name not in category_entries:
                category_entries[cat_name] = []
            category_entries[cat_name].append(
                f"[{entry.id}] {entry.content} (score: {score:.3f})"
            )

        for cat_name, entries in category_entries.items():
            section = f"=== {cat_name.upper()} ===\n" + "\n".join(entries)
            sections.append(section)

        return "\n\n".join(sections)

    def get_retrieval_summary(self, result: RetrievalResult) -> str:
        """Get human-readable summary of retrieval result"""
        lines = [
            f"Query: {result.query}",
            f"Selected Categories (top {self.k_categories}):"
        ]

        for category, score in result.selected_categories:
            lines.append(f"  - {category.value}: {score:.4f}")

        lines.append(f"\nRetrieved Entries: {result.total_retrieved}")

        for entry, score in result.retrieved_entries[:10]:  # Show top 10
            lines.append(f"  [{entry.id}] ({entry.category.value}) {entry.content[:50]}... | score: {score:.4f}")

        return "\n".join(lines)


class HybridRetriever(TwoLayerRAGRetriever):
    """
    Hybrid retriever that can combine BM25 with embedding-based retrieval
    """

    def __init__(self,
                 memory_system: MemorySystem,
                 k_categories: int = 3,
                 n_entries_per_category: int = 5,
                 use_core_memory: bool = True,
                 embedding_weight: float = 0.5):
        super().__init__(memory_system, k_categories, n_entries_per_category, use_core_memory)
        self.embedding_weight = embedding_weight
        self.bm25_weight = 1.0 - embedding_weight

    def compute_embedding_similarity(self,
                                      query_embedding: List[float],
                                      doc_embedding: List[float]) -> float:
        """Compute cosine similarity between embeddings"""
        if not query_embedding or not doc_embedding:
            return 0.0

        dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
        query_norm = math.sqrt(sum(a * a for a in query_embedding))
        doc_norm = math.sqrt(sum(a * a for a in doc_embedding))

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def hybrid_score(self,
                     bm25_score: float,
                     embedding_score: float,
                     max_bm25: float = 10.0) -> float:
        """Combine BM25 and embedding scores"""
        # Normalize BM25 score
        normalized_bm25 = min(bm25_score / max(max_bm25, 1), 1.0)
        # Embedding score is already in [0, 1] for cosine similarity
        return self.bm25_weight * normalized_bm25 + self.embedding_weight * embedding_score
