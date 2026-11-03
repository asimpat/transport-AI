from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.knowledge_document import KnowledgeDocument
from app.services.embedding_service import generate_embedding


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document content must not be empty.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must be a non-negative integer less than chunk_size.")

    text = text.strip()
    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


async def ingest_document(
    db: Session,
    title: str,
    content: str,
    source_type: str = "manual",
    source_reference: str | None = None,
) -> KnowledgeDocument:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Document title must not be empty.")

    chunks = split_text_into_chunks(content)
    try:
        document = KnowledgeDocument(
            title=title.strip(),
            content=content,
            source_type=source_type,
            source_reference=source_reference,
        )
        db.add(document)
        db.flush()

        for chunk_index, chunk_content in enumerate(chunks):
            embedding = await generate_embedding(chunk_content)
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    embedding=embedding,
                )
            )

        db.commit()
        return document
    except BaseException:
        db.rollback()
        raise