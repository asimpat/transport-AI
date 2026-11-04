import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.dialects import postgresql

from app.services.retrieval_service import retrieve_relevant_chunks


class RetrieveRelevantChunksTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.query = self.db.query.return_value
        self.query.join.return_value = self.query
        self.query.order_by.return_value = self.query
        self.query.limit.return_value = self.query
        self.embedding = [0.1] * 768
        self.generate_embedding = patch(
            "app.services.retrieval_service.generate_embedding",
            new_callable=AsyncMock,
            return_value=self.embedding,
        )
        self.mock_generate_embedding = self.generate_embedding.start()
        self.addCleanup(self.generate_embedding.stop)

    async def test_successful_retrieval_returns_document_and_chunk_fields(self):
        self.query.all.return_value = [
            SimpleNamespace(
                chunk_id=11,
                document_id=4,
                document_title="Lafia bus routes",
                chunk_index=2,
                content="Buses operate within Lafia.",
                source_type="manual",
                source_reference="transport guide",
                cosine_distance=0.15,
            )
        ]

        results = await retrieve_relevant_chunks(self.db, "Where do buses operate?")

        self.mock_generate_embedding.assert_awaited_once_with(
            "Where do buses operate?"
        )
        self.assertEqual(
            results,
            [
                {
                    "chunk_id": 11,
                    "document_id": 4,
                    "document_title": "Lafia bus routes",
                    "chunk_index": 2,
                    "content": "Buses operate within Lafia.",
                    "source_type": "manual",
                    "source_reference": "transport guide",
                    "similarity_score": 0.85,
                }
            ],
        )
        self.db.commit.assert_not_called()
        self.db.add.assert_not_called()
        self.db.delete.assert_not_called()

    async def test_top_k_is_used_as_query_limit(self):
        self.query.all.return_value = []

        results = await retrieve_relevant_chunks(
            self.db,
            "Where do buses operate?",
            top_k=3,
        )

        self.assertEqual(results, [])
        self.query.limit.assert_called_once_with(3)

    async def test_rejects_empty_question(self):
        for question in ("", " ", "\n\t"):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "Question must not be empty"):
                    await retrieve_relevant_chunks(self.db, question)

        self.mock_generate_embedding.assert_not_awaited()
        self.db.query.assert_not_called()

    async def test_rejects_invalid_top_k(self):
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k must be a positive integer"):
                    await retrieve_relevant_chunks(
                        self.db,
                        "Where do buses operate?",
                        top_k=top_k,
                    )

        self.mock_generate_embedding.assert_not_awaited()
        self.db.query.assert_not_called()

    async def test_orders_by_ascending_cosine_distance(self):
        near = SimpleNamespace(
            chunk_id=1,
            document_id=1,
            document_title="Near",
            chunk_index=0,
            content="Near match",
            source_type="manual",
            source_reference=None,
            cosine_distance=0.1,
        )
        far = SimpleNamespace(
            chunk_id=2,
            document_id=2,
            document_title="Far",
            chunk_index=0,
            content="Far match",
            source_type="manual",
            source_reference=None,
            cosine_distance=0.7,
        )
        self.query.all.return_value = [near, far]

        results = await retrieve_relevant_chunks(self.db, "Question")

        self.assertEqual([result["chunk_id"] for result in results], [1, 2])
        order_expression = self.query.order_by.call_args.args[0]
        sql = str(order_expression.compile(dialect=postgresql.dialect()))
        self.assertIn("embedding <=>", sql)
        self.assertTrue(sql.endswith("ASC"))

    async def test_similarity_score_is_one_minus_cosine_distance(self):
        self.query.all.return_value = [
            SimpleNamespace(
                chunk_id=1,
                document_id=2,
                document_title="Source",
                chunk_index=0,
                content="Chunk",
                source_type="manual",
                source_reference=None,
                cosine_distance=0.32,
            )
        ]

        results = await retrieve_relevant_chunks(self.db, "Question")

        self.assertAlmostEqual(results[0]["similarity_score"], 0.68)

    async def test_empty_knowledge_base_returns_empty_list(self):
        self.query.all.return_value = []

        results = await retrieve_relevant_chunks(self.db, "Question")

        self.assertEqual(results, [])
        self.db.commit.assert_not_called()

    async def test_embedding_failure_is_propagated_without_querying_database(self):
        error = RuntimeError("Embedding service unavailable")
        self.mock_generate_embedding.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "Embedding service unavailable"):
            await retrieve_relevant_chunks(self.db, "Question")

        self.db.query.assert_not_called()
        self.db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
