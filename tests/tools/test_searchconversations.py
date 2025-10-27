"""
SearchConversations Tool Tests

Tests for the searchconversations tool including parameter parsing,
search functionality, sorting, output formatting, and error handling.
"""

import sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.searchconversations import searchconversations


# Mock data generators
def create_mock_result(
    chat_id: str = "abc123",
    timestamp: float = None,
    score: float = 10.0,
    message_count: int = 25,
    preview: str = "Sample conversation",
    search_snippet: str = "Sample snippet",
) -> dict:
    """Create a mock search result."""
    if timestamp is None:
        timestamp = datetime.now().timestamp()

    return {
        "id": chat_id,
        "timestamp": timestamp,
        "score": score,
        "message_count": message_count,
        "preview": preview,
        "search_snippet": search_snippet,
    }


def test_searchconversations_parameter_parsing_valid():
    """Test valid parameter parsing."""
    xml_input = """<query>docker deployment</query>
    <top_k>5</top_k>
    <order_by>best_match</order_by>"""

    params = searchconversations._parse_parameters(xml_input)

    assert params["query"] == "docker deployment"
    assert params["top_k"] == 5
    assert params["order_by"] == "best_match"


def test_searchconversations_parameter_parsing_defaults():
    """Test parameter parsing with defaults."""
    xml_input = "<query>simple search</query>"

    params = searchconversations._parse_parameters(xml_input)

    assert params["query"] == "simple search"
    assert params["top_k"] == 10  # default
    assert params["order_by"] == "best_match"  # default


def test_searchconversations_parameter_parsing_missing_query():
    """Test that missing query raises error."""
    xml_input = "<top_k>5</top_k>"

    try:
        searchconversations._parse_parameters(xml_input)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "query" in str(e).lower()


def test_searchconversations_parameter_parsing_empty_query():
    """Test that empty query raises error."""
    xml_input = "<query></query>"

    try:
        searchconversations._parse_parameters(xml_input)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_searchconversations_parameter_parsing_invalid_top_k():
    """Test invalid top_k values."""
    # Zero top_k
    xml_input = "<query>test</query><top_k>0</top_k>"
    try:
        searchconversations._parse_parameters(xml_input)
        assert False, "Should have raised ValueError for zero top_k"
    except ValueError as e:
        assert "positive" in str(e).lower()

    # Negative top_k
    xml_input = "<query>test</query><top_k>-5</top_k>"
    try:
        searchconversations._parse_parameters(xml_input)
        assert False, "Should have raised ValueError for negative top_k"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_searchconversations_parameter_parsing_top_k_capping():
    """Test that excessive top_k is capped at 100."""
    xml_input = "<query>test</query><top_k>150</top_k>"

    params = searchconversations._parse_parameters(xml_input)

    assert params["top_k"] == 100


def test_searchconversations_parameter_parsing_invalid_order_by():
    """Test invalid order_by value."""
    xml_input = "<query>test</query><order_by>invalid_order</order_by>"

    try:
        searchconversations._parse_parameters(xml_input)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "order_by" in str(e).lower()


def test_searchconversations_parameter_parsing_order_by_case_insensitive():
    """Test that order_by parsing is case insensitive."""
    xml_input = "<query>test</query><order_by>NEWEST</order_by>"

    params = searchconversations._parse_parameters(xml_input)

    assert params["order_by"] == "newest"


@patch("core.chat.Chat")
def test_searchconversations_search_success(mock_chat_class):
    """Test successful search."""
    # Setup mock
    mock_results = [
        create_mock_result("abc123", score=15.0),
        create_mock_result("def456", score=10.0),
    ]
    mock_chat_class.search_chats.return_value = mock_results

    # Execute
    results = searchconversations._search_conversations("test query", top_k=10)

    # Verify
    assert len(results) == 2
    assert results[0]["id"] == "abc123"
    assert results[1]["id"] == "def456"
    mock_chat_class.search_chats.assert_called_once_with("test query", top_k=10)


@patch("core.chat.Chat")
def test_searchconversations_search_empty_results(mock_chat_class):
    """Test search with no results."""
    mock_chat_class.search_chats.return_value = []

    results = searchconversations._search_conversations("nonexistent query")

    assert results == []


@patch("core.chat.Chat")
def test_searchconversations_search_error_handling(mock_chat_class):
    """Test error handling during search."""
    mock_chat_class.search_chats.side_effect = Exception("Database error")

    try:
        searchconversations._search_conversations("test query")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Database error" in str(e)


def test_searchconversations_sorting_best_match():
    """Test sorting by best_match (relevance score)."""
    results = [
        create_mock_result("c1", score=5.0),
        create_mock_result("c2", score=15.0),
        create_mock_result("c3", score=10.0),
    ]

    sorted_results = searchconversations._apply_sorting(results, "best_match")

    assert sorted_results[0]["id"] == "c2"  # score 15.0
    assert sorted_results[1]["id"] == "c3"  # score 10.0
    assert sorted_results[2]["id"] == "c1"  # score 5.0


def test_searchconversations_sorting_newest():
    """Test sorting by newest (most recent timestamp)."""
    now = datetime.now().timestamp()
    one_day_ago = (datetime.now() - timedelta(days=1)).timestamp()
    two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()

    results = [
        create_mock_result("c1", timestamp=two_days_ago),
        create_mock_result("c2", timestamp=now),
        create_mock_result("c3", timestamp=one_day_ago),
    ]

    sorted_results = searchconversations._apply_sorting(results, "newest")

    assert sorted_results[0]["id"] == "c2"  # most recent
    assert sorted_results[1]["id"] == "c3"  # middle
    assert sorted_results[2]["id"] == "c1"  # oldest


def test_searchconversations_sorting_oldest():
    """Test sorting by oldest."""
    now = datetime.now().timestamp()
    one_day_ago = (datetime.now() - timedelta(days=1)).timestamp()
    two_days_ago = (datetime.now() - timedelta(days=2)).timestamp()

    results = [
        create_mock_result("c1", timestamp=now),
        create_mock_result("c2", timestamp=two_days_ago),
        create_mock_result("c3", timestamp=one_day_ago),
    ]

    sorted_results = searchconversations._apply_sorting(results, "oldest")

    assert sorted_results[0]["id"] == "c2"  # oldest
    assert sorted_results[1]["id"] == "c3"  # middle
    assert sorted_results[2]["id"] == "c1"  # newest


def test_searchconversations_format_result():
    """Test result formatting."""
    result = create_mock_result(
        chat_id="abc123",
        timestamp=datetime(2025, 10, 25, 14, 30, 0).timestamp(),
        score=12.5,
        message_count=47,
        preview="Testing docker deployment",
        search_snippet="container configuration snippet",
    )

    formatted = searchconversations._format_result(result, 1)

    assert "abc123" in formatted
    assert "2025-10-25 14:30:00" in formatted
    assert "47" in formatted
    assert "12.50" in formatted
    assert "Testing docker deployment" in formatted
    assert "container configuration snippet" in formatted


def test_searchconversations_format_result_long_preview():
    """Test that long previews are truncated."""
    long_preview = "x" * 100
    result = create_mock_result(preview=long_preview)

    formatted = searchconversations._format_result(result, 1)

    # Should contain truncation indicator
    assert "..." in formatted
    # Should not contain full preview
    assert long_preview not in formatted


def test_searchconversations_format_output_with_results():
    """Test output formatting with results."""
    results = [
        create_mock_result("c1", score=15.0, preview="First chat"),
        create_mock_result("c2", score=10.0, preview="Second chat"),
    ]

    output = searchconversations._format_output(results, "test query", "best_match")

    assert "Found 2 archived conversation(s)" in output
    assert "test query" in output
    assert "best_match" in output
    assert "First chat" in output
    assert "Second chat" in output


def test_searchconversations_format_output_no_results():
    """Test output formatting with no results."""
    output = searchconversations._format_output([], "test query", "best_match")

    assert "No archived conversations found" in output
    assert "test query" in output


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_run_success(mock_search):
    """Test successful tool execution."""
    # Setup
    mock_results = [create_mock_result("c1", score=15.0)]
    mock_search.return_value = mock_results

    xml_input = """<query>test</query>
    <top_k>10</top_k>
    <order_by>best_match</order_by>"""

    # Execute
    result = searchconversations.run(xml_input)

    # Verify
    assert "Found 1 archived conversation(s)" in result
    assert "test" in result


def test_searchconversations_run_missing_query():
    """Test tool execution with missing query."""
    xml_input = "<top_k>5</top_k>"

    result = searchconversations.run(xml_input)

    assert "Error" in result
    assert "query" in result.lower()


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_run_error_handling(mock_search):
    """Test tool error handling."""
    mock_search.side_effect = Exception("Search failed")

    xml_input = "<query>test</query>"

    result = searchconversations.run(xml_input)

    assert "Error" in result
    assert "Failed to search conversations" in result


def test_searchconversations_get_delim():
    """Test tool delimiter."""
    assert searchconversations.get_delim() == "searchconversations"


def test_searchconversations_get_tool_info():
    """Test tool info."""
    info = searchconversations.get_tool_info()

    assert "name" in info
    assert info["name"] == "searchconversations"
    assert "description" in info
    assert "example" in info
    assert len(info["description"]) > 0


def test_searchconversations_tool_info_contains_example():
    """Test that tool info example is valid XML."""
    info = searchconversations.get_tool_info()
    example = info["example"]

    assert "<searchconversations>" in example
    assert "</searchconversations>" in example
    assert "<query>" in example
    assert "<top_k>" in example
    assert "<order_by>" in example


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_complex_query(mock_search):
    """Test handling of complex search queries."""
    # Complex query with special characters
    complex_query = "docker-compose && traefik | (kubernetes OR k8s)"

    mock_search.return_value = []

    xml_input = f"<query>{complex_query}</query>"
    result = searchconversations.run(xml_input)

    # Should not error on complex query
    assert "No archived conversations" in result
    mock_search.assert_called_once()


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_whitespace_handling(mock_search):
    """Test handling of queries with various whitespace."""
    mock_search.return_value = []

    # Query with leading/trailing whitespace
    xml_input = """<query>

    docker deployment

    </query>"""

    result = searchconversations.run(xml_input)

    # Should handle whitespace correctly
    assert "No archived conversations" in result
    # Verify that whitespace was stripped
    call_args = mock_search.call_args[0]
    assert call_args[0] == "docker deployment"


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_all_sort_orders(mock_search):
    """Test all three sort order options."""
    results = [create_mock_result(f"c{i}") for i in range(3)]
    mock_search.return_value = results

    for order in ["best_match", "newest", "oldest"]:
        xml_input = f"<query>test</query><order_by>{order}</order_by>"
        result = searchconversations.run(xml_input)

        assert "Error" not in result
        assert f"sorted by {order}" in result


@patch("tools.searchconversations.searchconversations._search_conversations")
def test_searchconversations_various_top_k_values(mock_search):
    """Test various top_k parameter values."""
    # Return variable number of results
    mock_search.side_effect = [
        [create_mock_result(f"c{i}") for i in range(5)],  # 5 results
        [create_mock_result(f"c{i}") for i in range(20)],  # 20 results (capped at top_k)
        [],  # 0 results
    ]

    # top_k = 5
    xml_input = "<query>test1</query><top_k>5</top_k>"
    result1 = searchconversations.run(xml_input)
    assert "Found 5" in result1

    # top_k = 20 (should work)
    xml_input = "<query>test2</query><top_k>20</top_k>"
    result2 = searchconversations.run(xml_input)
    assert "Found 20" in result2

    # top_k = 1
    xml_input = "<query>test3</query><top_k>1</top_k>"
    mock_search.return_value = []
    result3 = searchconversations.run(xml_input)
    assert "No archived conversations" in result3


# Test level assignments
test_searchconversations_parameter_parsing_valid.level = 2
test_searchconversations_parameter_parsing_defaults.level = 2
test_searchconversations_parameter_parsing_missing_query.level = 2
test_searchconversations_parameter_parsing_empty_query.level = 2
test_searchconversations_parameter_parsing_invalid_top_k.level = 2
test_searchconversations_parameter_parsing_top_k_capping.level = 2
test_searchconversations_parameter_parsing_invalid_order_by.level = 2
test_searchconversations_parameter_parsing_order_by_case_insensitive.level = 2
test_searchconversations_search_success.level = 2
test_searchconversations_search_empty_results.level = 2
test_searchconversations_search_error_handling.level = 2
test_searchconversations_sorting_best_match.level = 2
test_searchconversations_sorting_newest.level = 2
test_searchconversations_sorting_oldest.level = 2
test_searchconversations_format_result.level = 2
test_searchconversations_format_result_long_preview.level = 2
test_searchconversations_format_output_with_results.level = 2
test_searchconversations_format_output_no_results.level = 2
test_searchconversations_run_success.level = 2
test_searchconversations_run_missing_query.level = 2
test_searchconversations_run_error_handling.level = 2
test_searchconversations_get_delim.level = 2
test_searchconversations_get_tool_info.level = 2
test_searchconversations_tool_info_contains_example.level = 2
test_searchconversations_complex_query.level = 3
test_searchconversations_whitespace_handling.level = 3
test_searchconversations_all_sort_orders.level = 3
test_searchconversations_various_top_k_values.level = 3
