import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.core.config import settings
from app.services.generation_service import SYSTEM_PROMPT, generate_answer


class GenerateAnswerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.answer = "Buses operate within Lafia."
        self.response = httpx.Response(
            200,
            json={"message": {"content": self.answer}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )
        self.client = AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.client.post.return_value = self.response
        self.async_client_patch = patch(
            "app.services.generation_service.httpx.AsyncClient",
            return_value=self.client,
        )
        self.mock_async_client = self.async_client_patch.start()
        self.addCleanup(self.async_client_patch.stop)

    async def test_successful_response_returns_assistant_content(self):
        answer = await generate_answer(
            "Where do buses operate?",
            "Buses operate within Lafia.",
        )

        self.assertEqual(answer, self.answer)

    async def test_uses_configured_chat_model_and_request_payload(self):
        await generate_answer("Where do buses operate?", "Buses operate within Lafia.")

        self.mock_async_client.assert_called_once_with(
            timeout=settings.OLLAMA_TIMEOUT_SECONDS
        )
        self.client.post.assert_awaited_once_with(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_CHAT_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Retrieved context:\nBuses operate within Lafia."
                        "\n\nQuestion:\nWhere do buses operate?",
                    },
                ],
            },
        )
        self.assertEqual(settings.OLLAMA_CHAT_MODEL, "qwen2.5:1.5b")

    async def test_rejects_empty_question(self):
        for question in ("", " ", "\n\t"):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "Question must not be empty"):
                    await generate_answer(question, "Some context")

        self.client.post.assert_not_awaited()

    async def test_empty_context_prompts_for_knowledge_base_fallback(self):
        await generate_answer("Where do buses operate?", " \n")

        payload = self.client.post.await_args.kwargs["json"]
        user_message = payload["messages"][1]["content"]
        self.assertIn("No retrieved context is available", user_message)
        self.assertIn("not available in the current knowledge base", user_message)
        self.assertIn("Where do buses operate?", user_message)

    async def test_http_error_is_propagated(self):
        self.client.post.return_value = httpx.Response(
            500,
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        with self.assertRaises(httpx.HTTPStatusError):
            await generate_answer("Question", "Context")

    async def test_connection_error_is_propagated(self):
        self.client.post.side_effect = httpx.ConnectError("Ollama is unavailable")

        with self.assertRaises(httpx.ConnectError):
            await generate_answer("Question", "Context")

    async def test_invalid_response_is_rejected(self):
        self.client.post.return_value = httpx.Response(
            200,
            json={"message": {"content": None}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        with self.assertRaisesRegex(ValueError, "message.content"):
            await generate_answer("Question", "Context")

    async def test_invalid_json_is_reported(self):
        self.client.post.return_value = httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            await generate_answer("Question", "Context")


if __name__ == "__main__":
    unittest.main()
