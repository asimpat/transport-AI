import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app


class ChatEndpointTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()

        def override_get_db():
            yield self.db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)
        self.rag_patch = patch(
            "app.api.v1.endpoints.chat.ask_transport_question",
            new_callable=AsyncMock,
            return_value={
                "question": "Where do buses operate?",
                "answer": "Buses operate within Lafia.",
                "sources": [
                    {
                        "source_label": "[1]",
                        "chunk_id": 11,
                        "document_id": 4,
                        "document_title": "Lafia bus routes",
                        "source_type": "manual",
                        "source_reference": "transport guide",
                        "similarity_score": 0.92,
                    }
                ],
                "retrieved_chunks_count": 1,
            },
        )
        self.ask_transport_question = self.rag_patch.start()
        self.addCleanup(self.rag_patch.stop)
        self.addCleanup(app.dependency_overrides.clear)

    def test_successful_submission_returns_valid_response_structure(self):
        response = self.client.post(
            "/api/v1/chat/ask",
            json={"question": "Where do buses operate?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "question": "Where do buses operate?",
                "answer": "Buses operate within Lafia.",
                "sources": [
                    {
                        "source_label": "[1]",
                        "chunk_id": 11,
                        "document_id": 4,
                        "document_title": "Lafia bus routes",
                        "source_type": "manual",
                        "source_reference": "transport guide",
                        "similarity_score": 0.92,
                    }
                ],
                "retrieved_chunks_count": 1,
            },
        )
        self.ask_transport_question.assert_awaited_once_with(
            self.db,
            "Where do buses operate?",
        )

    def test_empty_question_is_rejected(self):
        response = self.client.post("/api/v1/chat/ask", json={"question": ""})

        self.assertEqual(response.status_code, 422)
        self.ask_transport_question.assert_not_awaited()

    def test_whitespace_question_is_rejected(self):
        response = self.client.post("/api/v1/chat/ask", json={"question": " \n\t"})

        self.assertEqual(response.status_code, 422)
        self.ask_transport_question.assert_not_awaited()

    def test_question_longer_than_2000_characters_is_rejected(self):
        response = self.client.post(
            "/api/v1/chat/ask",
            json={"question": "q" * 2001},
        )

        self.assertEqual(response.status_code, 422)
        self.ask_transport_question.assert_not_awaited()

    def test_rag_failure_returns_generic_internal_server_error(self):
        self.ask_transport_question.side_effect = RuntimeError("secret internals")

        response = self.client.post(
            "/api/v1/chat/ask",
            json={"question": "Where do buses operate?"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Unable to process the question at this time."},
        )
        self.assertNotIn("secret internals", response.text)

    def test_sources_are_returned_without_metadata_loss(self):
        response = self.client.post(
            "/api/v1/chat/ask",
            json={"question": "Where do buses operate?"},
        )

        source = response.json()["sources"][0]
        self.assertEqual(source["source_label"], "[1]")
        self.assertEqual(source["chunk_id"], 11)
        self.assertEqual(source["document_id"], 4)
        self.assertEqual(source["document_title"], "Lafia bus routes")
        self.assertEqual(source["source_type"], "manual")
        self.assertEqual(source["source_reference"], "transport guide")
        self.assertEqual(source["similarity_score"], 0.92)


if __name__ == "__main__":
    unittest.main()
