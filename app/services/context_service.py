def build_rag_context(retrieved_chunks: list[dict]) -> dict:
    context_parts = []
    sources = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        source_label = f"[{source_number}]"
        context_parts.append(
            f"{source_label} {chunk['document_title']}\n{chunk['content']}"
        )
        sources.append(
            {
                "source_label": source_label,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "source_type": chunk["source_type"],
                "source_reference": chunk["source_reference"],
                "similarity_score": chunk["similarity_score"],
            }
        )

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }
