import re

from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.knowledge_document import KnowledgeDocument
from app.services.embedding_service import generate_embedding


MIN_SIMILARITY_SCORE = 0.50
TRANSPORT_RECORD_ID_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d{3}\b",
    re.IGNORECASE,
)


def _query_chunk_rows(db: Session, cosine_distance):
    return db.query(
        DocumentChunk.id.label("chunk_id"),
        DocumentChunk.document_id.label("document_id"),
        KnowledgeDocument.title.label("document_title"),
        DocumentChunk.chunk_index.label("chunk_index"),
        DocumentChunk.content.label("content"),
        KnowledgeDocument.source_type.label("source_type"),
        KnowledgeDocument.source_reference.label("source_reference"),
        cosine_distance.label("cosine_distance"),
    ).join(
        KnowledgeDocument,
        DocumentChunk.document_id == KnowledgeDocument.id,
    )


def _has_exact_identifier(content: str, identifier: str) -> bool:
    return re.search(
        rf"(?<![A-Z0-9-]){re.escape(identifier)}(?![A-Z0-9-])",
        content,
        re.IGNORECASE,
    ) is not None


def _is_record_heading(content: str, identifier: str) -> bool:
    return re.search(
        rf"(?im)^\s*#{{1,6}}\s*{re.escape(identifier)}(?=\s|:|$)",
        content,
    ) is not None


def _row_to_result(row) -> dict[str, object]:
    return {
        "chunk_id": row.chunk_id,
        "document_id": row.document_id,
        "document_title": row.document_title,
        "chunk_index": row.chunk_index,
        "content": row.content,
        "source_type": row.source_type,
        "source_reference": row.source_reference,
        "similarity_score": 1 - float(row.cosine_distance),
    }


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
        _query_chunk_rows(db, cosine_distance)
        .filter(cosine_distance <= 1 - MIN_SIMILARITY_SCORE)
        .order_by(cosine_distance.asc())
        .limit(top_k)
        .all()
    )

    results = [_row_to_result(row) for row in rows]
    identifier_match = TRANSPORT_RECORD_ID_PATTERN.search(question)
    if identifier_match is None:
        return results

    identifier = identifier_match.group(0)
    exact_rows = (
        _query_chunk_rows(db, cosine_distance)
        .filter(DocumentChunk.content.ilike(f"%{identifier}%"))
        .all()
    )
    exact_record_results = [
        _row_to_result(row)
        for row in exact_rows
        if _has_exact_identifier(row.content, identifier)
        and _is_record_heading(row.content, identifier)
    ]
    exact_record_results.sort(key=lambda result: result["similarity_score"], reverse=True)

    exact_chunk_ids = {result["chunk_id"] for result in exact_record_results}
    remaining_results = [
        result
        for result in results
        if result["chunk_id"] not in exact_chunk_ids
        and not (
            _has_exact_identifier(result["content"], identifier)
            and not _is_record_heading(result["content"], identifier)
        )
    ]
    return (exact_record_results + remaining_results)[:top_k]
