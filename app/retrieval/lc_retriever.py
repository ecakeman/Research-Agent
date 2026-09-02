from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.models.schemas import RetrievedChunk
from app.retrieval.hybrid import HybridRetriever


class LangChainHybridRetriever(BaseRetriever):
    """LangChain Retriever 抽象；内部仍走自研 HybridRetriever。"""

    hybrid: HybridRetriever
    top_k: int = 20

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.hybrid.search(query, self.top_k)
        return [chunk_to_document(h) for h in hits]


def chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "score": chunk.score,
            "title": chunk.source_title,
            "url": chunk.source_url,
            "heading_path": chunk.heading_path,
        },
    )
