import httpx

from app.core.config import settings

EMBEDDING_DIMENSIONS = 768


async def generate_embedding(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Text to embed must not be empty.")

    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url,
            json={
                "model": settings.OLLAMA_EMBEDDING_MODEL,
                "prompt": text,
            },
        )

        response.raise_for_status()
        data = response.json()

    embedding = data.get("embedding") if isinstance(data, dict) else None
    if (
        not isinstance(embedding, list)
        or len(embedding) != EMBEDDING_DIMENSIONS
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in embedding
        )
    ):
        raise ValueError(
            "Ollama returned an invalid embedding; expected "
            f"{EMBEDDING_DIMENSIONS} numeric values."
        )

    return [float(value) for value in embedding]
