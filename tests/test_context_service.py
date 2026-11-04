import unittest

from app.services.context_service import build_rag_context


class BuildRagContextTests(unittest.TestCase):
    def setUp(self):
        self.first_chunk = {
            "chunk_id": 11,
            "document_id": 4,
            "document_title": "Lafia bus routes",
            "chunk_index": 0,
            "content": "Buses operate within Lafia.",
            "source_type": "manual",
            "source_reference": "transport guide",
            "similarity_score": 0.92,
        }
        self.second_chunk = {
            "chunk_id": 12,
            "document_id": 5,
            "document_title": "Lafia fares",
            "chunk_index": 1,
            "content": "Fare information is not listed.",
            "source_type": "official",
            "source_reference": None,
            "similarity_score": 0.76,
        }

    def test_empty_chunks_return_empty_context_and_sources(self):
        self.assertEqual(
            build_rag_context([]),
            {"context": "", "sources": []},
        )

    def test_one_chunk_formats_context_and_preserves_source_metadata(self):
        result = build_rag_context([self.first_chunk])

        self.assertEqual(
            result["context"],
            "[1] Lafia bus routes\nBuses operate within Lafia.",
        )
        self.assertEqual(
            result["sources"],
            [
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
        )

    def test_multiple_chunks_use_sequential_labels_and_preserve_content(self):
        result = build_rag_context([self.first_chunk, self.second_chunk])

        self.assertEqual(
            result["context"],
            "[1] Lafia bus routes\nBuses operate within Lafia.\n\n"
            "[2] Lafia fares\nFare information is not listed.",
        )
        self.assertIn("Lafia bus routes", result["context"])
        self.assertIn("Lafia fares", result["context"])
        self.assertIn(self.first_chunk["content"], result["context"])
        self.assertIn(self.second_chunk["content"], result["context"])
        self.assertEqual(
            [source["source_label"] for source in result["sources"]],
            ["[1]", "[2]"],
        )

    def test_similarity_scores_are_preserved(self):
        result = build_rag_context([self.first_chunk, self.second_chunk])

        self.assertEqual(
            [source["similarity_score"] for source in result["sources"]],
            [0.92, 0.76],
        )


if __name__ == "__main__":
    unittest.main()
