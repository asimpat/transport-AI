import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.models.document_chunk import DocumentChunk
from app.models.knowledge_document import KnowledgeDocument
from app.services.document_service import ingest_document, split_text_into_chunks


class SplitTextIntoChunksTests(unittest.TestCase):
    def test_short_text_produces_one_chunk(self):
        self.assertEqual(
            split_text_into_chunks("Short transport information."),
            ["Short transport information."],
        )

    def test_long_text_produces_multiple_chunks(self):
        chunks = split_text_into_chunks("a" * 2500, chunk_size=1000, chunk_overlap=150)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))

    def test_chunks_preserve_requested_overlap(self):
        chunks = split_text_into_chunks("abcdefghij", chunk_size=6, chunk_overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_empty_content_is_rejected(self):
        for content in ("", " ", "\n\t"):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, "content must not be empty"):
                    split_text_into_chunks(content)

    def test_invalid_chunk_size_is_rejected(self):
        for chunk_size in (0, -1, 1.5, True):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    split_text_into_chunks("content", chunk_size=chunk_size)

    def test_invalid_chunk_overlap_is_rejected(self):
        for chunk_overlap in (-1, 5, 1.5, True):
            with self.subTest(chunk_overlap=chunk_overlap):
                with self.assertRaisesRegex(ValueError, "chunk_overlap"):
                    split_text_into_chunks(
                        "content",
                        chunk_size=5,
                        chunk_overlap=chunk_overlap,
                    )


class IngestDocumentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db.flush.side_effect = lambda: setattr(
            self.db.add.call_args.args[0], "id", 42
        )

    async def test_successful_ingestion_creates_document_and_chunks(self):
        content = "a" * 1100
        expected_chunks = [content[:1000], content[850:]]
        embeddings = [[0.1] * 768, [0.2] * 768]

        with patch(
            "app.services.document_service.generate_embedding",
            new_callable=AsyncMock,
            side_effect=embeddings,
        ) as generate_embedding:
            document = await ingest_document(
                self.db,
                title="  Lafia transport  ",
                content=content,
                source_reference="operator notes",
            )

        self.assertIsInstance(document, KnowledgeDocument)
        self.assertEqual(document.id, 42)
        self.assertEqual(document.title, "Lafia transport")
        self.assertEqual(document.content, content)
        self.assertEqual(document.source_type, "manual")
        self.assertEqual(document.source_reference, "operator notes")
        generate_embedding.assert_has_awaits(
            [call(expected_chunks[0]), call(expected_chunks[1])]
        )

        added = [item.args[0] for item in self.db.add.call_args_list]
        self.assertIsInstance(added[0], KnowledgeDocument)
        chunks = added[1:]
        self.assertEqual(len(chunks), 2)
        for index, chunk in enumerate(chunks):
            with self.subTest(chunk_index=index):
                self.assertIsInstance(chunk, DocumentChunk)
                self.assertEqual(chunk.document_id, 42)
                self.assertEqual(chunk.chunk_index, index)
                self.assertEqual(chunk.content, expected_chunks[index])
                self.assertEqual(chunk.embedding, embeddings[index])

        self.db.commit.assert_called_once_with()
        self.db.rollback.assert_not_called()

    async def test_embedding_failure_rolls_back_transaction(self):
        error = RuntimeError("Embedding service unavailable")

        with patch(
            "app.services.document_service.generate_embedding",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "Embedding service unavailable"):
                await ingest_document(
                    self.db,
                    title="Lafia transport",
                    content="Transport details",
                )

        self.db.commit.assert_not_called()
        self.db.rollback.assert_called_once_with()
