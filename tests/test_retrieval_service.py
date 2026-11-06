import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.dialects import postgresql

from app.services.retrieval_service import (
    MIN_SIMILARITY_SCORE,
    retrieve_relevant_chunks,
)


def retrieval_row(**fields):
    row = {
        "chunk_id": 1,
        "document_id": 1,
        "document_title": "Demo document",
        "chunk_index": 0,
        "content": "Demo content",
        "source_type": "manual",
        "source_reference": None,
        "cosine_distance": 0.1,
    }
    return SimpleNamespace(**(row | fields))


def exact_identifier_row(content, **fields):
    return retrieval_row(**({"content": content, "cosine_distance": 0.1} | fields))


class RetrieveRelevantChunksTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.query = self.db.query.return_value
        self.query.join.return_value = self.query
        self.query.filter.return_value = self.query
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
            retrieval_row(
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

    async def test_exact_identifier_prioritizes_record_over_test_question_mention(self):
        semantic_test_list = retrieval_row(
            chunk_id=21,
            content="1. What is the destination of TEST-KEKE-001?",
            cosine_distance=0.05,
        )
        semantic_other = retrieval_row(
            chunk_id=22,
            document_title="Other route",
            content="Unrelated semantic match.",
            cosine_distance=0.2,
        )
        actual_record = exact_identifier_row(
            "## TEST-KEKE-001: Demo Stop North to Demo Stop South\n"
            "- Origin: Demo Stop North\n"
            "- Intermediate stops: Demo Stop Cedar, Demo Stop Lantern",
            chunk_id=23,
            document_title="SIMULATED DATA - Synthetic Keke Routes",
            source_type="synthetic",
            source_reference="knowledge_base/synthetic/synthetic_keke_routes.md",
            cosine_distance=0.4,
        )
        test_question = exact_identifier_row(
            "1. What is the destination of TEST-KEKE-001?",
            chunk_id=24,
            cosine_distance=0.1,
        )
        self.query.all.side_effect = [
            [semantic_test_list, semantic_other],
            [test_question, actual_record],
        ]

        results = await retrieve_relevant_chunks(
            self.db,
            "What is TEST-KEKE-001, and which stops does it use?",
        )

        self.assertEqual(results[0]["chunk_id"], 23)
        self.assertIn("Demo Stop Cedar", results[0]["content"])
        self.assertEqual(results[0]["source_type"], "synthetic")
        self.assertNotIn("destination of TEST-KEKE-001", results[1]["content"])
        self.assertEqual(self.query.all.call_count, 2)

    async def test_exact_identifier_search_requires_a_full_identifier_match(self):
        exact_partial_record = exact_identifier_row(
            "## TEST-KEKE-0010: A different record",
            chunk_id=31,
        )
        self.query.all.side_effect = [[], [exact_partial_record]]

        results = await retrieve_relevant_chunks(
            self.db,
            "Tell me about TEST-KEKE-001",
        )

        self.assertEqual(results, [])

    async def test_questions_without_identifier_use_normal_semantic_retrieval(self):
        self.query.all.return_value = [retrieval_row(chunk_id=41)]

        results = await retrieve_relevant_chunks(self.db, "Where do buses operate?")

        self.assertEqual([result["chunk_id"] for result in results], [41])
        self.query.all.assert_called_once_with()

    async def test_top_k_is_used_as_query_limit(self):
        self.query.all.return_value = []

        results = await retrieve_relevant_chunks(
            self.db,
            "Where do buses operate?",
            top_k=3,
        )

        self.assertEqual(results, [])
        self.query.limit.assert_called_once_with(3)

    async def test_similarity_filter_includes_above_and_exact_threshold(self):
        above_threshold = retrieval_row(
            chunk_id=1,
            document_id=1,
            document_title="Above threshold",
            chunk_index=0,
            content="Above",
            source_type="manual",
            source_reference=None,
            cosine_distance=1 - (MIN_SIMILARITY_SCORE + 0.1),
        )
        exact_threshold = retrieval_row(
            chunk_id=2,
            document_id=2,
            document_title="At threshold",
            chunk_index=0,
            content="At threshold",
            source_type="manual",
            source_reference=None,
            cosine_distance=1 - MIN_SIMILARITY_SCORE,
        )
        below_threshold = retrieval_row(
            chunk_id=3,
            document_id=3,
            document_title="Below threshold",
            chunk_index=0,
            content="Below",
            source_type="manual",
            source_reference=None,
            cosine_distance=1 - (MIN_SIMILARITY_SCORE - 0.1),
        )

        self.query.all.return_value = [
            above_threshold,
            exact_threshold,
        ]

        results = await retrieve_relevant_chunks(self.db, "Question")

        self.assertEqual([item["chunk_id"] for item in results], [1, 2])
        predicate = self.query.filter.call_args_list[0].args[0]
        compiled = predicate.compile(dialect=postgresql.dialect())
        self.assertIn("<=", str(compiled))
        self.assertIn(1 - MIN_SIMILARITY_SCORE, compiled.params.values())

    async def test_similarity_filter_excludes_below_threshold_only(self):
        below_threshold = retrieval_row(
            chunk_id=3,
            document_id=3,
            document_title="Below threshold",
            chunk_index=0,
            content="Below",
            source_type="manual",
            source_reference=None,
            cosine_distance=1 - (MIN_SIMILARITY_SCORE - 0.01),
        )
        self.query.all.return_value = []

        results = await retrieve_relevant_chunks(self.db, "Question")

        self.assertEqual(results, [])
        predicate = self.query.filter.call_args_list[0].args[0]
        compiled = predicate.compile(dialect=postgresql.dialect())
        cutoff = next(
            value
            for value in compiled.params.values()
            if isinstance(value, (int, float))
        )
        self.assertLess(
            1 - below_threshold.cosine_distance,
            MIN_SIMILARITY_SCORE,
        )
        self.assertGreater(below_threshold.cosine_distance, cutoff)
        self.assertEqual(cutoff, 1 - MIN_SIMILARITY_SCORE)

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
        near = retrieval_row(
            chunk_id=1,
            document_id=1,
            document_title="Near",
            chunk_index=0,
            content="Near match",
            source_type="manual",
            source_reference=None,
            cosine_distance=0.1,
        )
        far = retrieval_row(
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
            retrieval_row(
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
