"""Destination Guide RAG Tool.

Retrieves section-coherent chunks from the destination knowledge base.
"""

from typing import Optional
import logging
from langchain_core.tools import tool

from config.settings import settings
from vectorstore.store import get_vector_store

logger = logging.getLogger("tripmate.tools.rag")


@tool
def search_destination_guide(query: str) -> list[str]:
    """Search destination guides for verified travel information.

    Use this tool for questions about visas, entry requirements, passport rules,
    best time to visit, seasonal highlights, local customs, cultural etiquette,
    tipping, packing tips, clothing advice, or safety and health for supported
    destinations (Tokyo, Bangkok, Barcelona, Reykjavik).
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to search_destination_guide.")
        return ["Error: Search query cannot be empty. Please specify a destination or topic."]

    clean_query = query.strip()
    logger.info("Executing search_destination_guide with query: '%s'", clean_query)

    try:
        store = get_vector_store()
        chunks = store.similarity_search(clean_query, k=settings.TOP_K_CHUNKS)

        if not chunks:
            supported = ", ".join(settings.SUPPORTED_CITIES)
            return [
                f"No destination guide information found for '{clean_query}'. "
                f"Currently supported destinations in TripMate are: {supported}."
            ]

        results = [
            f"[{chunk.city.upper()} | {chunk.section}]\n{chunk.content}"
            for chunk in chunks
        ]
        return results

    except Exception as exc:
        logger.error("Error during search_destination_guide: %s", exc, exc_info=True)
        return [f"Unable to retrieve destination guide at this moment due to an internal error: {str(exc)}"]
