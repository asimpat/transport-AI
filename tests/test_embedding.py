import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.core.config import settings
from app.services.embedding_service import generate_embedding


class GenerateEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_configured_request_and_returns_vector(self):
        expected_embedding = [0.25] * 768
        response = httpx.Response(
            200,
            json={"embedding": expected_embedding},
            request=httpx.Request("POST", "http://localhost:11434/api/embeddings"),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response

        with patch(
            "app.services.embedding_service.httpx.AsyncClient",
            return_value=client,
        ) as async_client:
            embedding = await generate_embedding("Commercial buses operate within Lafia.")

        self.assertEqual(embedding, expected_embedding)
        async_client.assert_called_once_with(timeout=settings.OLLAMA_TIMEOUT_SECONDS)
        client.post.assert_awaited_once_with(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
            json={
                "model": settings.OLLAMA_EMBEDDING_MODEL,
                "prompt": "Commercial buses operate within Lafia.",
            },
        )

    async def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            await generate_embedding("  ")

    async def test_rejects_invalid_embedding_dimensions(self):
        response = httpx.Response(
            200,
            json={"embedding": [0.25]},
            request=httpx.Request("POST", "http://localhost:11434/api/embeddings"),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response

        with patch(
            "app.services.embedding_service.httpx.AsyncClient",
            return_value=client,
        ):
            with self.assertRaisesRegex(ValueError, "768 numeric values"):
                await generate_embedding("Commercial buses operate within Lafia.")


async def main() -> None:
    text = "Commercial buses operate within Lafia."

    embedding = await generate_embedding(text)

    print("Embedding generated successfully!")
    print("Vector dimensions:", len(embedding))
    print("First five values:", embedding[:5])


if __name__ == "__main__":
    asyncio.run(main())
