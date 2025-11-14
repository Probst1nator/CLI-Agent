# tests/tools/test_paper_search.py
"""
Tests for the paper_search tool, particularly focusing on query handling.
"""
import pytest
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.paper_search import paper_search


class TestPaperSearchQueryBuilding:
    """Test the _build_arxiv_query method to ensure proper query formatting."""

    def test_simple_query_unchanged(self):
        """Simple queries without field prefixes should remain unchanged."""
        query = paper_search._build_arxiv_query("artificial intelligence")
        assert query == "artificial intelligence"

    def test_simple_query_with_spaces(self):
        """Simple queries with multiple words should remain unchanged."""
        query = paper_search._build_arxiv_query("generative AI models")
        assert query == "generative AI models"

    def test_query_with_field_prefix_unchanged(self):
        """Queries with field prefixes should remain unchanged."""
        query = paper_search._build_arxiv_query("ti:transformer")
        assert query == "ti:transformer"

    def test_query_with_author_field_unchanged(self):
        """Queries with author field should remain unchanged."""
        query = paper_search._build_arxiv_query("au:Hinton")
        assert query == "au:Hinton"

    def test_query_with_abstract_field_unchanged(self):
        """Queries with abstract field should remain unchanged."""
        query = paper_search._build_arxiv_query("abs:deep learning")
        assert query == "abs:deep learning"

    def test_query_with_category_field_unchanged(self):
        """Queries with category field should remain unchanged."""
        query = paper_search._build_arxiv_query("cat:cs.AI")
        assert query == "cat:cs.AI"

    def test_query_with_boolean_operators_unchanged(self):
        """Queries with boolean operators should remain unchanged."""
        query = paper_search._build_arxiv_query("transformer AND attention")
        assert query == "transformer AND attention"

    def test_query_with_or_operator_unchanged(self):
        """Queries with OR operator should remain unchanged."""
        query = paper_search._build_arxiv_query("transformer OR attention")
        assert query == "transformer OR attention"

    def test_query_with_andnot_operator_unchanged(self):
        """Queries with ANDNOT operator should remain unchanged."""
        query = paper_search._build_arxiv_query("machine learning ANDNOT survey")
        assert query == "machine learning ANDNOT survey"

    def test_complex_query_with_fields_unchanged(self):
        """Complex queries with field prefixes and boolean operators unchanged."""
        query = paper_search._build_arxiv_query("ti:transformer AND au:Vaswani")
        assert query == "ti:transformer AND au:Vaswani"

    def test_simple_query_with_author_filter(self):
        """Simple query with author filter should add both."""
        query = paper_search._build_arxiv_query("deep learning", author="Hinton")
        assert query == 'deep learning AND au:"Hinton"'

    def test_simple_query_with_category_filter(self):
        """Simple query with category filter should add both."""
        query = paper_search._build_arxiv_query("neural networks", category="cs.LG")
        assert query == "neural networks AND cat:cs.LG"

    def test_simple_query_with_both_filters(self):
        """Simple query with author and category filters should add all."""
        query = paper_search._build_arxiv_query(
            "attention mechanism",
            author="Vaswani",
            category="cs.CL"
        )
        assert query == 'attention mechanism AND au:"Vaswani" AND cat:cs.CL'

    def test_field_query_with_category_filter(self):
        """Query with field prefix and category filter should add category."""
        query = paper_search._build_arxiv_query("ti:transformer", category="cs.AI")
        assert query == "ti:transformer AND cat:cs.AI"

    def test_empty_query(self):
        """Empty query should return empty string."""
        query = paper_search._build_arxiv_query("")
        assert query == ""

    def test_query_case_insensitive_field_detection(self):
        """Field prefix detection should be case-insensitive."""
        query = paper_search._build_arxiv_query("TI:transformer")
        assert query == "TI:transformer"

    def test_reported_issue_queries(self):
        """Test the specific queries mentioned in the bug report."""
        # These should remain unchanged (arXiv API handles default field searching)
        queries = [
            "artificial intelligence",
            "LLM",
            "Generative AI"
        ]

        for q in queries:
            result = paper_search._build_arxiv_query(q)
            assert result == q, f"Query '{q}' should remain unchanged"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
