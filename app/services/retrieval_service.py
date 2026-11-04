from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.knowledge_document import KnowledgeDocument
from app.services.embedding_service import generate_embedding


MIN_SIMILARITY_SCORE = 0.50


async def retrieve_relevant_chunks(
    db: Session,
    question: str,
    top_k: int = 5,
) -> list[dict[str, object]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    question_embedding = await generate_embedding(question)
    cosine_distance = DocumentChunk.embedding.cosine_distance(question_embedding)

    rows = (
        db.query(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id.label("document_id"),
            KnowledgeDocument.title.label("document_title"),
            DocumentChunk.chunk_index.label("chunk_index"),
            DocumentChunk.content.label("content"),
            KnowledgeDocument.source_type.label("source_type"),
            KnowledgeDocument.source_reference.label("source_reference"),
            cosine_distance.label("cosine_distance"),
        )
        .join(
            KnowledgeDocument,
            DocumentChunk.document_id == KnowledgeDocument.id,
        )
        .filter(cosine_distance <= 1 - MIN_SIMILARITY_SCORE)
        .order_by(cosine_distance.asc())
        .limit(top_k)
        .all()
    )

    return [
        {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "document_title": row.document_title,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "source_type": row.source_type,
            "source_reference": row.source_reference,
            "similarity_score": 1 - float(row.cosine_distance),
        }
        for row in rows
    ]
