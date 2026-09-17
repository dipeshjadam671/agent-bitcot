"""In-memory Vector Store for Destination Guides with OpenAI Embeddings.

Provides swappable vector storage and cosine similarity retrieval across
destination guide section chunks. Includes city and section boosting.
"""

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Any, Callable, Optional
import logging

from config.settings import settings
from vectorstore.ingest import DocumentChunk, load_raw_destination_files

logger = logging.getLogger("tripmate.vectorstore")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors in pure Python."""
    if len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Stores to ensure swappability."""

    @abstractmethod
    def add_documents(self, documents: list[DocumentChunk]) -> None:
        """Embed and store document chunks."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        city: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Retrieve the top-k most similar document chunks."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store using OpenAI embeddings and cosine similarity."""

    def __init__(
        self,
        embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ):
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []
        self._custom_embedder = embedder
        self._is_initialized = False

    def _get_embeddings_client(self):
        """Lazy load OpenAIEmbeddings from langchain_openai."""
        if self._custom_embedder is not None:
            return None
        from langchain_openai import OpenAIEmbeddings

        # If API key is not valid (e.g. testing), let it fall back or raise informative error
        api_key = settings.OPENAI_API_KEY or "sk-dummy"
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=api_key,
            max_retries=1,
            request_timeout=5,
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings."""
        if self._custom_embedder is not None:
            return self._custom_embedder(texts)

        if not settings.validate_openai_key():
            logger.info("No valid OpenAI API key detected. Using deterministic lexical embeddings.")
            return [self._fallback_lexical_vector(t) for t in texts]

        client = self._get_embeddings_client()
        try:
            return client.embed_documents(texts)
        except Exception as exc:
            logger.warning(
                "OpenAI embeddings call failed (%s). Falling back to deterministic lexical vectors.",
                exc,
            )
            return [self._fallback_lexical_vector(t) for t in texts]

    def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single search query."""
        if self._custom_embedder is not None:
            return self._custom_embedder([query])[0]

        if not settings.validate_openai_key():
            return self._fallback_lexical_vector(query)

        client = self._get_embeddings_client()
        try:
            return client.embed_query(query)
        except Exception as exc:
            logger.warning(
                "OpenAI query embedding failed (%s). Falling back to deterministic lexical vector.",
                exc,
            )
            return self._fallback_lexical_vector(query)

    @staticmethod
    def _fallback_lexical_vector(text: str, dim: int = 256) -> list[float]:
        """Deterministic hash-based lexical vector used for offline testing or fallback."""
        vec = [0.0] * dim
        tokens = text.lower().split()
        for token in tokens:
            idx = abs(hash(token)) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def add_documents(self, documents: list[DocumentChunk]) -> None:
        """Embed and index document chunks."""
        if not documents:
            return

        texts_to_embed = [doc.formatted_text for doc in documents]
        embeddings = self._embed_texts(texts_to_embed)

        self._chunks.extend(documents)
        self._embeddings.extend(embeddings)
        self._is_initialized = True
        logger.info("Successfully indexed %d chunks in vector store.", len(documents))

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        city: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Perform cosine similarity search with optional city and section filtering / boosting."""
        if not self._chunks:
            return []

        query_vec = self._embed_query(query)
        query_lower = query.lower()

        # Automatic city context detection if not explicitly passed
        target_city = city.lower() if city else None
        if not target_city:
            for supported_city in settings.SUPPORTED_CITIES:
                if supported_city.lower() in query_lower:
                    target_city = supported_city.lower()
                    break

        # Automatic section intent detection if not explicitly passed
        target_section = section.upper() if section else None
        if not target_section:
            if any(term in query_lower for term in ["pack", "clothes", "wear", "shoes", "luggage", "bring"]):
                target_section = "PACKING TIPS"
            elif any(term in query_lower for term in ["visa", "passport", "entry", "border", "stay"]):
                target_section = "VISA & ENTRY"
            elif any(term in query_lower for term in ["when", "best time", "season", "month to visit"]):
                target_section = "BEST TIME TO VISIT"
            elif any(term in query_lower for term in ["custom", "etiquette", "culture", "tip", "tipping", "temple"]):
                target_section = "LOCAL CUSTOMS"
            elif any(term in query_lower for term in ["safe", "health", "water", "crime", "emergency", "doctor"]):
                target_section = "SAFETY & HEALTH"

        scored_chunks: list[tuple[float, DocumentChunk]] = []

        for chunk, doc_vec in zip(self._chunks, self._embeddings):
            raw_sim = cosine_similarity(query_vec, doc_vec)
            score = raw_sim

            # Apply domain-specific semantic boosting
            if target_city and target_city in chunk.city.lower():
                score += 0.35  # Strong city alignment boost
            elif target_city and target_city not in chunk.city.lower():
                score -= 0.30  # Deprioritize irrelevant cities

            if target_section and target_section in chunk.section.upper():
                score += 0.25  # Relevant section boost

            # For packing queries, also boost BEST TIME TO VISIT for weather context
            if target_section == "PACKING TIPS" and "BEST TIME TO VISIT" in chunk.section.upper():
                score += 0.15

            scored_chunks.append((score, chunk))

        # Sort descending by final score
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_results = [chunk for _, chunk in scored_chunks[:k]]
        return top_results

    def load_and_index(self, raw_dir: Optional[Path] = None) -> int:
        """Load and index all raw destination guide files."""
        chunks = load_raw_destination_files(raw_dir)
        self.add_documents(chunks)
        return len(chunks)


# Module-level singleton
_vector_store_instance: Optional[InMemoryVectorStore] = None


def get_vector_store(
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
    force_reload: bool = False,
) -> InMemoryVectorStore:
    """Retrieve or initialize the singleton InMemoryVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None or force_reload or embedder is not None:
        store = InMemoryVectorStore(embedder=embedder)
        store.load_and_index()
        _vector_store_instance = store
    return _vector_store_instance
