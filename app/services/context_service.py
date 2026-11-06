def build_rag_context(retrieved_chunks: list[dict]) -> dict:
    context_parts = []
    sources = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        source_label = f"[{source_number}]"
        source_reference = chunk["source_reference"] or "Not provided"
        context_parts.append(
            f"--- BEGIN RETRIEVED CHUNK {source_label} ---\n"
            f"Document title: {chunk['document_title']}\n"
            f"Classification (source_type): {chunk['source_type']}\n"
            f"Source reference: {source_reference}\n"
            f"Chunk index: {chunk['chunk_index']}\n"
            f"Content:\n{chunk['content']}\n"
            f"--- END RETRIEVED CHUNK {source_label} ---"
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
