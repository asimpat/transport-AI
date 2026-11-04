import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag_service import (
    INSUFFICIENT_INFORMATION_ANSWER,
    ask_transport_question,
)


class AskTransportQuestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.question = "Where do buses operate?"
        self.retrieved_chunks = [
            {
                "chunk_id": 11,
                "document_id": 4,
                "document_title": "Lafia bus routes",
                "chunk_index": 0,
                "content": "Buses operate within Lafia.",
                "source_type": "manual",
                "source_reference": "transport guide",
                "similarity_score": 0.92,
            },
            {
                "chunk_id": 12,
                "document_id": 5,
                "document_title": "Lafia transport stops",
                "chunk_index": 1,
                "content": "Transport stops are listed in the guide.",
                "source_type": "official",
                "source_reference": None,
                "similarity_score": 0.81,
            },
        ]
        self.sources = [
            {
                "source_label": "[1]",
                "chunk_id": 11,
                "document_id": 4,
                "document_title": "Lafia bus routes",
                "source_type": "manual",
                "source_reference": "transport guide",
                "similarity_score": 0.92,
            }
        ]
        self.retrieve_patch = patch(
            "app.services.rag_service.retrieve_relevant_chunks",
            new_callable=AsyncMock,
            return_value=self.retrieved_chunks,
        )
        self.context_patch = patch(
            "app.services.rag_service.build_rag_context",
            return_value={
                "context": "[1] Lafia bus routes\nBuses operate within Lafia.",
                "sources": self.sources,
            },
        )
        self.generation_patch = patch(
            "app.services.rag_service.generate_answer",
            new_callable=AsyncMock,
            return_value="Buses operate within Lafia.",
        )
        self.retrieve = self.retrieve_patch.start()
        self.build_context = self.context_patch.start()
        self.generate = self.generation_patch.start()
        self.addCleanup(self.retrieve_patch.stop)
        self.addCleanup(self.context_patch.stop)
        self.addCleanup(self.generation_patch.stop)

    async def test_successful_rag_flow(self):
        result = await ask_transport_question(self.db, self.question)

        self.assertEqual(
            result,
            {
                "question": self.question,
                "answer": "Buses operate within Lafia.",
                "sources": self.sources,
                "retrieved_chunks_count": 2,
            },
        )
        self.retrieve.assert_awaited_once_with(self.db, self.question)
        self.build_context.assert_called_once_with(self.retrieved_chunks)
        self.generate.assert_awaited_once_with(
            self.question,
            "[1] Lafia bus routes\nBuses operate within Lafia.",
        )

    async def test_passes_question_unchanged_to_retrieval_and_generation(self):
        question = "  Where do buses operate?  "

        await ask_transport_question(self.db, question)

        self.retrieve.assert_awaited_once_with(self.db, question)
        self.generate.assert_awaited_once_with(
            question,
            "[1] Lafia bus routes\nBuses operate within Lafia.",
        )

    async def test_preserves_sources_and_retrieved_chunk_count(self):
        result = await ask_transport_question(self.db, self.question)

        self.assertIs(result["sources"], self.sources)
        self.assertEqual(result["retrieved_chunks_count"], len(self.retrieved_chunks))

    async def test_empty_question_is_rejected_before_pipeline_calls(self):
        for question in ("", " ", "\n\t"):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "Question must not be empty"):
                    await ask_transport_question(self.db, question)

        self.retrieve.assert_not_awaited()
        self.build_context.assert_not_called()
        self.generate.assert_not_awaited()

    async def test_no_retrieved_chunks_returns_deterministic_fallback(self):
        self.retrieve.return_value = []

        result = await ask_transport_question(self.db, self.question)

        self.build_context.assert_not_called()
        self.generate.assert_not_awaited()
        self.assertEqual(result["question"], self.question)
        self.assertEqual(result["answer"], INSUFFICIENT_INFORMATION_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks_count"], 0)

    async def test_relevant_chunks_still_call_context_and_generation(self):
        result = await ask_transport_question(self.db, self.question)

        self.build_context.assert_called_once_with(self.retrieved_chunks)
        self.generate.assert_awaited_once_with(
            self.question,
            "[1] Lafia bus routes\nBuses operate within Lafia.",
        )
        self.assertEqual(result["answer"], "Buses operate within Lafia.")
        self.assertEqual(result["sources"], self.sources)
        self.assertEqual(result["retrieved_chunks_count"], 2)

    async def test_retrieval_failure_is_propagated(self):
        error = RuntimeError("Retrieval failed")
        self.retrieve.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "Retrieval failed"):
            await ask_transport_question(self.db, self.question)

        self.build_context.assert_not_called()
        self.generate.assert_not_awaited()

    async def test_generation_failure_is_propagated(self):
        error = RuntimeError("Generation failed")
        self.generate.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "Generation failed"):
            await ask_transport_question(self.db, self.question)

        self.retrieve.assert_awaited_once_with(self.db, self.question)
        self.build_context.assert_called_once_with(self.retrieved_chunks)


if __name__ == "__main__":
    unittest.main()
