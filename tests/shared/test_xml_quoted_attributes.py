"""
Test for XML parsing with quoted attributes across tag boundaries.

This test specifically covers the bug fix for _mask_quoted_xml_tags() which was
incorrectly matching across XML tag boundaries, causing false positives.

Bug example: In <editfile> with <find>"value"</find><replace>"value"</replace>,
the old regex pattern r'"[^"]*<[a-zA-Z0-9_/]+[^"]*>"' would match the string
'"</find>\n  <replace>sort="' as a quoted tag, causing valid XML to be flagged.

Fix: Changed pattern to r'"[^"<]*<[a-zA-Z0-9_/]+>[^"<]*"' which excludes '<'
characters before and after the tag name, preventing cross-tag matches.
"""

from shared.common_utils import extract_blocks


def test_editfile_with_quoted_attributes():
    """
    Test the exact case reported as broken - editfile with quoted find/replace.

    This was causing false positive "XML syntax errors" because the masking
    function was matching across tag boundaries.
    """
    content = '''<editfile>
  <filepath>huggingface_api.py</filepath>
  <find>sort="trending"</find>
  <replace>sort="downloads"</replace>
</editfile>'''

    blocks = extract_blocks(content, ['editfile'])

    assert len(blocks) == 1
    # extract_blocks returns a list of strings containing the extracted content
    assert 'huggingface_api.py' in blocks[0]
    assert 'sort="trending"' in blocks[0]
    assert 'sort="downloads"' in blocks[0]


test_editfile_with_quoted_attributes.level = 3


def test_no_cross_tag_masking():
    """
    Test that quoted values at tag boundaries don't get masked together.

    The pattern "value1"</param1>\n<param2>"value2" should NOT be matched
    as a single quoted XML tag.
    """
    content = '''<tool>
  <param1>"value1"</param1>
  <param2>"value2"</param2>
  <param3>"value3"</param3>
</tool>'''

    blocks = extract_blocks(content, ['tool'])

    assert len(blocks) == 1
    # All quoted values should be preserved
    assert '"value1"' in blocks[0]
    assert '"value2"' in blocks[0]
    assert '"value3"' in blocks[0]


test_no_cross_tag_masking.level = 3


def test_quoted_single_tag_references_are_masked():
    """
    Test that legitimate quoted XML tag references ARE properly masked.

    Strings like "<bash>" or "<command>" in quotes should be masked to prevent
    them from being parsed as actual tool calls.
    """
    content = '''<bash>
  echo "Use the <grep> tool"
  echo 'Try <search> instead'
  echo `Run <command>`
</bash>'''

    blocks = extract_blocks(content, ['bash'])

    assert len(blocks) == 1
    # These quoted single-tag references should be preserved
    assert '<grep>' in blocks[0]
    assert '<search>' in blocks[0]
    assert '<command>' in blocks[0]


test_quoted_single_tag_references_are_masked.level = 3


def test_comparison_operators_in_quoted_attributes():
    """
    Test that comparison operators (< >) in quoted attribute values work correctly.

    This ensures the fix doesn't break legitimate uses of < and > in attribute values.
    Note: XML parser escapes < and > to &lt; and &gt; when they appear in content.
    """
    content = '''<editfile>
  <filepath>test.py</filepath>
  <find>if value < 10:</find>
  <replace>if value > 10:</replace>
</editfile>'''

    blocks = extract_blocks(content, ['editfile'])

    assert len(blocks) == 1
    # XML parser escapes < and > in content
    assert 'value &lt; 10' in blocks[0]
    assert 'value &gt; 10' in blocks[0]


test_comparison_operators_in_quoted_attributes.level = 3


def test_multiple_quoted_strings_in_sequence():
    """
    Test multiple lines with quoted strings don't interfere with each other.
    """
    content = '''<bash>
  echo "first value"
  echo "second value"
  echo "third value"
</bash>'''

    blocks = extract_blocks(content, ['bash'])

    assert len(blocks) == 1
    assert 'first value' in blocks[0]
    assert 'second value' in blocks[0]
    assert 'third value' in blocks[0]


test_multiple_quoted_strings_in_sequence.level = 3


if __name__ == '__main__':
    print("Testing XML masking bug fix for quoted attributes...\n")

    test_editfile_with_quoted_attributes()
    print("✓ Editfile with quoted attributes")

    test_no_cross_tag_masking()
    print("✓ No cross-tag masking")

    test_quoted_single_tag_references_are_masked()
    print("✓ Quoted single tags are masked")

    test_comparison_operators_in_quoted_attributes()
    print("✓ Comparison operators in attributes")

    test_multiple_quoted_strings_in_sequence()
    print("✓ Multiple quoted strings")

    print("\n✅ All quoted attribute tests passed!")
