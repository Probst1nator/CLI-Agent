#!/usr/bin/env python3
"""
Comprehensive edge case tests for XML parsing infrastructure.
Tests all corner cases, special characters, and real-world scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.common_utils import extract_blocks
import pytest


class TestSpecialCharactersAndEscaping:
    """Test Category 1: Special Characters & Escaping"""

    def test_multiple_comparison_operators(self):
        """Multiple comparison operators in LaTeX"""
        xml = """<tool>
Content with $a < b > c <= d >= e$ and more.
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "$a < b > c <= d >= e$" in blocks[0]

    def test_nested_angle_brackets(self):
        """Nested angle brackets (should work with escaping)"""
        xml = """<tool>
Text with &lt;&lt;nested&gt;&gt; brackets.
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "<<nested>>" in blocks[0]  # Should be unescaped after parsing

    def test_ampersands(self):
        """Various ampersand patterns"""
        xml = """<tool>
A &amp; B, regular & ampersand, and logical &&.
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        # Note: &amp; in input stays as &amp; because our escaper double-escapes it
        # Input &amp; → escaped to &amp;amp; → parsed to &amp; → extracted as &amp;
        # Regular & → escaped to &amp; → parsed to & → extracted as &
        # This is expected: pre-escaped entities stay escaped, raw & gets escaped
        assert "&amp; B" in blocks[0]
        assert "regular & ampersand" in blocks[0]

    def test_quotes_in_content(self):
        """Double and single quotes"""
        xml = """<tool>
Text with "double quotes" and 'single quotes' and mixed "quote's".
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert '"double quotes"' in blocks[0]
        assert "'single quotes'" in blocks[0]

    def test_mixed_escaped_unescaped(self):
        """Mixed escaped and unescaped special chars"""
        xml = """<tool>
$r &lt; r_0 < r_1 > r_2 &gt; r_3$
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1


class TestTagPatterns:
    """Test Category 2: Tag Patterns"""

    def test_self_closing_tags(self):
        """Self-closing tags (may not extract content)"""
        xml = """<tool/>"""
        blocks = extract_blocks(xml, "tool")
        # Self-closing tags might return empty content or no blocks
        # This documents the behavior
        assert len(blocks) <= 1

    def test_empty_tags(self):
        """Empty tags with no content"""
        xml = """<tool></tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert blocks[0] == "" or blocks[0].strip() == ""

    def test_tags_with_attributes(self):
        """Tags with XML attributes"""
        xml = """<tool arg="value" flag="true">
Content here
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "Content here" in blocks[0]

    def test_nested_same_name_tags(self):
        """Nested tags with same name (edge case)"""
        xml = """<tool>
Outer content
<tool>Inner content</tool>
More outer
</tool>"""
        blocks = extract_blocks(xml, "tool")
        # This is invalid XML structure - nested same-name tags cannot parse correctly
        # The regex matches but ET.fromstring fails, so we log warning and skip
        # This documents that this edge case is handled gracefully (no crash)
        assert len(blocks) == 0  # Expected: skipped due to parse error

    def test_whitespace_in_tags(self):
        """Unusual whitespace patterns in tags"""
        # Note: < tool > is invalid XML, but test with valid spacing
        xml = """<tool  >
Content with tag spacing
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1


class TestMultipleBlocks:
    """Test Category 3: Multiple Blocks"""

    def test_three_sequential_identical_tags(self):
        """Three or more identical tags in sequence"""
        xml = """<bash>cmd1</bash><bash>cmd2</bash><bash>cmd3</bash>"""
        blocks = extract_blocks(xml, "bash")
        assert len(blocks) == 3
        assert blocks[0].strip() == "cmd1"
        assert blocks[1].strip() == "cmd2"
        assert blocks[2].strip() == "cmd3"

    def test_interleaved_different_tags(self):
        """Different tool types interleaved"""
        xml = """<bash>ls</bash><readfile>file.txt</readfile><bash>pwd</bash>"""
        bash_blocks = extract_blocks(xml, "bash")
        readfile_blocks = extract_blocks(xml, "readfile")
        assert len(bash_blocks) == 2
        assert len(readfile_blocks) == 1

    def test_duplicate_content_same_file(self):
        """Same file path in multiple tags"""
        xml = """<readfile>
./same_file.pdf
</readfile><readfile>
./same_file.pdf
</readfile>"""
        blocks = extract_blocks(xml, "readfile")
        assert len(blocks) == 2
        assert blocks[0].strip() == blocks[1].strip()

    def test_no_whitespace_between_tags(self):
        """Tags immediately adjacent (already tested but confirming)"""
        xml = """<tool>content1</tool><tool>content2</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 2


class TestComplexContent:
    """Test Category 4: Complex Content"""

    def test_code_blocks_within_xml(self):
        """Markdown code blocks inside XML"""
        xml = """<visualresponse>
  <type>markdown</type>
  <content>
# Code Example

```python
def example():
    return x > 5 and y < 10
```
  </content>
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1
        assert "```python" in blocks[0]
        # Note: When there are child elements (<type>, <content>),
        # ET.tostring() re-escapes special chars, so > becomes &gt;
        # This is expected behavior - tools receive the inner XML structure
        assert "x &gt; 5 and y &lt; 10" in blocks[0]

    def test_html_like_content(self):
        """HTML tags within tool content"""
        xml = """<tool>
Text with <div>HTML-like</div> content.
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        # Note: This might not parse correctly due to <div> being seen as XML
        # Document the behavior

    def test_urls_with_query_params(self):
        """URLs containing & in query strings"""
        xml = """<tool>
http://example.com?a=1&amp;b=2&amp;c=3
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        # &amp; should be decoded to &
        assert "a=1&b=2&c=3" in blocks[0]

    def test_complex_markdown_table(self):
        """Markdown tables with special formatting"""
        xml = """<visualresponse>
  <content>
| Header 1 | Header 2 |
| :--- | ---: |
| $x < 0$ | $y > 0$ |
| **bold** | *italic* |
  </content>
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1
        # When there are child elements, ET.tostring() escapes special chars
        assert "$x &lt; 0$" in blocks[0]
        assert "$y &gt; 0$" in blocks[0]

    def test_unicode_and_emoji(self):
        """Unicode characters and emoji"""
        xml = """<tool>
🚀 Rocket launch
✅ Success
❌ Failure
Überprüfung
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "🚀" in blocks[0]
        assert "✅" in blocks[0]
        assert "Überprüfung" in blocks[0]


class TestBoundaryCases:
    """Test Category 5: Boundary Cases"""

    def test_very_long_content(self):
        """Content exceeding 10KB"""
        long_content = "Line with $x < y$ content.\n" * 500  # ~15KB
        xml = f"""<tool>
{long_content}
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert len(blocks[0]) > 10000

    def test_empty_with_special_chars(self):
        """Empty-ish content with only special chars"""
        xml = """<tool>< ></tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1

    def test_only_whitespace(self):
        """Content with only whitespace"""
        xml = """<tool>
\t
  </tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert blocks[0].strip() == ""

    def test_special_chars_at_boundaries(self):
        """Special characters at start and end"""
        xml = """<tool>< content with specials ></tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1


class TestRealWorldToolCalls:
    """Test Category 6: Real-World Tool Calls"""

    def test_bash_with_pipes(self):
        """Bash commands with pipes"""
        xml = """<bash>ls -la | grep test | wc -l</bash>"""
        blocks = extract_blocks(xml, "bash")
        assert len(blocks) == 1
        assert "|" in blocks[0]
        assert "grep" in blocks[0]

    def test_bash_with_redirects(self):
        """Bash commands with input/output redirection"""
        xml = """<bash>cat file.txt > output.txt 2>&1</bash>"""
        blocks = extract_blocks(xml, "bash")
        assert len(blocks) == 1
        assert ">" in blocks[0]

    def test_bash_with_comparison(self):
        """Bash commands with numeric comparisons"""
        xml = """<bash>
if [ $count -gt 5 ]; then
    echo "greater"
fi
</bash>"""
        blocks = extract_blocks(xml, "bash")
        assert len(blocks) == 1
        assert "-gt 5" in blocks[0]

    def test_search_patterns_with_regex(self):
        """Search patterns containing regex special chars"""
        xml = """<grepsearch>
<pattern>.*error.*|.*warning.*</pattern>
</grepsearch>"""
        blocks = extract_blocks(xml, "grepsearch")
        assert len(blocks) == 1
        assert ".*error.*" in blocks[0]

    def test_nested_xml_structure(self):
        """Tool with nested XML structure"""
        xml = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
Text with $r > r_0$ comparison.
  </content>
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1
        # Should include nested tags
        assert "<action>" in blocks[0]
        assert "<type>" in blocks[0]


class TestMalformedXMLDetection:
    """Test that truly malformed XML is still caught"""

    def test_mismatched_tags(self):
        """Mismatched opening and closing tags"""
        xml = """<tool>content</wrong>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 0

    def test_visualresponse_with_nested_action_type_content(self):
        """Real-world case: visualresponse with properly nested action/type/content tags

        This was previously triggering a false positive in the pre-check regex which matched
        <visualresponse> with the first </action> closing tag, incorrectly reporting:
        "XML parsing failed: mismatched tag" when the XML was actually valid.
        """
        xml = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
Some content here
  </content>
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert "Some content here" in blocks[0]
        # Should also include the nested structure
        assert "<action>" in blocks[0]
        assert "<type>" in blocks[0]
        assert "<content>" in blocks[0]

    def test_actual_mismatched_opening_closing(self):
        """Opening tag doesn't match closing tag"""
        xml = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
</action>"""
        blocks = extract_blocks(xml, "visualresponse")
        # This truly malformed XML should be skipped
        assert len(blocks) == 0

    def test_unclosed_tag(self):
        """Tag opened but never closed"""
        xml = """<tool>content that never ends"""
        blocks = extract_blocks(xml, "tool")
        # Regex won't match unclosed tags
        assert len(blocks) == 0

    def test_invalid_tag_names(self):
        """Invalid characters in tag names"""
        xml = """<tool-name>content</tool-name>"""
        blocks = extract_blocks(xml, "tool-name")
        # Hyphens are actually valid in XML, should work
        # Test documents behavior


class TestExtractBlocksFiltering:
    """Test language/tag filtering in extract_blocks"""

    def test_filter_specific_tag(self):
        """Extract only specific tag when multiple present"""
        xml = """<bash>ls</bash><readfile>file.txt</readfile><python>print()</python>"""
        blocks = extract_blocks(xml, "readfile")
        assert len(blocks) == 1
        assert "file.txt" in blocks[0]

    def test_filter_with_list(self):
        """Extract multiple specific tags"""
        xml = """<bash>ls</bash><readfile>file.txt</readfile><python>print()</python>"""
        blocks = extract_blocks(xml, ["bash", "python"])
        assert len(blocks) == 2

    def test_extract_all_when_no_filter(self):
        """Extract all blocks when language=None"""
        xml = """<bash>ls</bash><readfile>file.txt</readfile>"""
        blocks = extract_blocks(xml, None)
        # Should extract both
        assert len(blocks) >= 2

    def test_case_insensitivity(self):
        """Tag names should be case-insensitive"""
        xml = """<Tool>content</Tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        blocks2 = extract_blocks(xml, "TOOL")
        assert len(blocks2) == 1


class TestTailTextExtraction:
    """Test Category 7: Tail Text Extraction (text after child elements)"""

    def test_tail_text_after_child_element(self):
        """Text that appears after a child element (tail text)"""
        xml = """<visualresponse>
  <action>create</action>
  Text after action tag
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1
        assert "Text after action tag" in blocks[0]

    def test_multiple_tail_texts(self):
        """Multiple child elements each with tail text"""
        xml = """<visualresponse>
  <action>create</action>
  First tail text
  <type>markdown</type>
  Second tail text
</visualresponse>"""
        blocks = extract_blocks(xml, "visualresponse")
        assert len(blocks) == 1
        assert "First tail text" in blocks[0]
        assert "Second tail text" in blocks[0]

    def test_mixed_root_text_children_and_tail(self):
        """Root text, child elements, and tail text combined"""
        xml = """<tool>
Root text before children
<arg1>value1</arg1>
Tail after arg1
<arg2>value2</arg2>
Tail after arg2
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "Root text before children" in blocks[0]
        assert "Tail after arg1" in blocks[0]
        assert "Tail after arg2" in blocks[0]

    def test_deeply_nested_with_tail_text(self):
        """Nested elements with tail text at multiple levels"""
        xml = """<tool>
<level1>L1 content</level1>
Tail L1
<level2>
  <inner>Inner content</inner>
  Tail inner
</level2>
Final tail
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "Tail L1" in blocks[0]
        assert "Tail inner" in blocks[0]
        assert "Final tail" in blocks[0]


class TestDoubleEscaping:
    """Test Category 8: Prevent Double-Escaping"""

    def test_already_escaped_ampersand(self):
        """Pre-escaped &amp; should not become &amp;amp;"""
        xml = """<tool>A &amp; B</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert blocks[0] == "A & B"

    def test_mixed_escaped_and_raw_special_chars(self):
        """Combination of pre-escaped and raw special characters"""
        xml = """<tool>Pre-escaped &lt; and raw < comparison</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1

    def test_multiple_escaping_passes(self):
        """Content that might be processed multiple times"""
        xml = """<tool>$r &lt; r_0$ and $r < r_1$</tool>"""
        blocks1 = extract_blocks(xml, "tool")
        blocks2 = extract_blocks(xml, "tool")
        assert blocks1 == blocks2


class TestEmptyAndWhitespaceChildren:
    """Test Category 9: Empty and Whitespace Child Elements"""

    def test_empty_child_elements(self):
        """Empty child elements mixed with content"""
        xml = """<tool>
<empty></empty>
Content here
<also_empty/>
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "Content here" in blocks[0]

    def test_whitespace_only_children(self):
        """Child elements containing only whitespace"""
        xml = """<tool>
<whitespace>
  </whitespace>
Real content
</tool>"""
        blocks = extract_blocks(xml, "tool")
        assert len(blocks) == 1
        assert "Real content" in blocks[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
