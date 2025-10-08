#!/usr/bin/env python3
"""
Test XML parsing infrastructure to diagnose the visualresponse XML validation issue.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.common_utils import extract_blocks
import xml.etree.ElementTree as ET


def test_case_1_basic_visualresponse():
    """Test 1: Basic visualresponse with simple content"""
    xml = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
Simple content without special characters.
  </content>
</visualresponse>"""

    print("\n=== Test 1: Basic visualresponse ===")
    print(f"Input: {xml[:100]}...")
    try:
        blocks = extract_blocks(xml, "visualresponse")
        print(f"✅ SUCCESS: Extracted {len(blocks)} blocks")
        if blocks:
            print(f"Content preview: {blocks[0][:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_case_2_latex_with_comparison():
    """Test 2: Content with LaTeX math containing < and >"""
    xml = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
The energy is zero when $r > r_0$ and negative when $r < r_0$.
  </content>
</visualresponse>"""

    print("\n=== Test 2: LaTeX with < and > ===")
    print(f"Input: {xml[:150]}...")
    try:
        blocks = extract_blocks(xml, "visualresponse")
        print(f"✅ SUCCESS: Extracted {len(blocks)} blocks")
        if blocks:
            print(f"Content preview: {blocks[0][:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_case_3_actual_failing_content():
    """Test 3: The actual content from the log that failed"""
    xml = r"""<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
Das ist ein zentrales Konzept in der Werkstoffkunde! Hier ist die Erklärung:

### Erklärung: Bindungspotenzial und Thermische Ausdehnung

#### 1. Die Skizze des Bindungspotenzials

Die potenzielle Energie $U_{pot}$ zwischen zwei Atomen ist die Summe aus:

| Achse | Bedeutung |
| :--- | :--- |
| **Y-Achse** | Potenzielle Energie $U_{pot}$ |
| **X-Achse** | Atomabstand $r$ |

**Verlauf der Kurve:**

1.  **Große Abstände ($r \to \infty$):** Die Energie ist Null.
2.  **Mittlere Abstände (Anziehung):** Die Energie wird negativ (Bindung entsteht).
3.  **Minimum:** Am tiefsten Punkt der Kurve herrscht Gleichgewicht.
    *   **$r_0$ (Gleichgewichtsabstand):** Der Abstand, bei dem die anziehenden und abstoßenden Kräfte gleich groß sind.
    *   **$E_B$ (Bindungsenergie):** Die Tiefe des Minimums.
4.  **Kleine Abstände (Abstoßung):** Die Energie steigt steil an.

#### 2. Erklärung der Thermischen Ausdehnung

Die thermische Ausdehnung basiert auf der **Asymmetrie** der $U_{pot}$-Kurve.

1.  **Temperatur und Energie:** Bei steigender Temperatur nimmt die Schwingungsenergie der Atome zu.
2.  **Asymmetrie:** Die Kurve ist im Bereich der Abstoßung (kleine $r$) **steiler** als im Bereich der Anziehung (große $r$).
3.  **Folge:** Aufgrund dieser Asymmetrie führt eine Zunahme der Schwingungsenergie dazu, dass die Atome **weiter in den Bereich größerer Abstände ($r > r_0$) vordringen** als in den Bereich kleinerer Abstände ($r < r_0$).
4.  **Ergebnis:** Der **mittlere Atomabstand** verschiebt sich mit steigender Temperatur zu größeren Werten.

**Merke:** Je **flacher** und **asymmetrischer** die Potentialmulde, desto **größer** ist der thermische Ausdehnungskoeffizient.
  </content>
</visualresponse>"""

    print("\n=== Test 3: Actual failing content with $r > r_0$ and $r < r_0$ ===")
    print(f"Input length: {len(xml)} characters")
    print("Searching for problematic characters...")

    # Find all < and > occurrences
    import re
    less_than = [(m.start(), m.group()) for m in re.finditer(r'<', xml)]
    greater_than = [(m.start(), m.group()) for m in re.finditer(r'>', xml)]

    print(f"Found {len(less_than)} '<' characters")
    print(f"Found {len(greater_than)} '>' characters")

    # Find specifically the ones in LaTeX
    latex_comparisons = re.findall(r'\$r\s*[<>]\s*r_0\$', xml)
    print(f"Found {len(latex_comparisons)} LaTeX comparison expressions: {latex_comparisons}")

    try:
        blocks = extract_blocks(xml, "visualresponse")
        print(f"✅ SUCCESS: Extracted {len(blocks)} blocks")
        if blocks:
            print(f"Content length: {len(blocks[0])} characters")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"Error type: {type(e).__name__}")


def test_case_4_direct_xml_parse():
    """Test 4: Direct XML parsing with ElementTree"""
    xml_with_special_chars = """<content>
The energy is zero when $r > r_0$ and negative when $r < r_0$.
</content>"""

    print("\n=== Test 4: Direct ET.fromstring with < and > ===")
    print(f"Input: {xml_with_special_chars}")
    try:
        root = ET.fromstring(xml_with_special_chars)
        print("✅ SUCCESS: Parsed successfully")
        print(f"Text: {root.text}")
    except ET.ParseError as e:
        print(f"❌ FAILED: {e}")
        print(f"Error details: line {e.position[0]}, column {e.position[1]}")


def test_case_5_escaped_xml():
    """Test 5: Properly escaped XML"""
    xml_escaped = """<content>
The energy is zero when $r &gt; r_0$ and negative when $r &lt; r_0$.
</content>"""

    print("\n=== Test 5: Properly escaped with &gt; and &lt; ===")
    print(f"Input: {xml_escaped}")
    try:
        root = ET.fromstring(xml_escaped)
        print("✅ SUCCESS: Parsed successfully")
        print(f"Text: {root.text}")
        print(f"Note: Entities automatically decoded to: {repr(root.text)}")
    except ET.ParseError as e:
        print(f"❌ FAILED: {e}")


def test_case_6_cdata_section():
    """Test 6: Using CDATA section"""
    xml_cdata = """<content><![CDATA[
The energy is zero when $r > r_0$ and negative when $r < r_0$.
]]></content>"""

    print("\n=== Test 6: Using CDATA section ===")
    print(f"Input: {xml_cdata}")
    try:
        root = ET.fromstring(xml_cdata)
        print("✅ SUCCESS: Parsed successfully")
        print(f"Text: {root.text}")
    except ET.ParseError as e:
        print(f"❌ FAILED: {e}")


def test_case_7_extract_blocks_behavior():
    """Test 7: How extract_blocks handles malformed XML"""
    malformed_xml = """Some text before
<visualresponse>
  <content>Text with $r < r_0$ comparison</content>
</visualresponse>
Some text after"""

    print("\n=== Test 7: extract_blocks behavior with malformed XML ===")
    print(f"Input: {malformed_xml[:100]}...")

    blocks = extract_blocks(malformed_xml, "visualresponse")
    print(f"Extracted {len(blocks)} blocks")
    if blocks:
        for i, block in enumerate(blocks):
            print(f"Block {i+1}: {block[:100]}...")


def test_case_8_duplicate_tags():
    """Test 8: Multiple identical tags in sequence (regression test for tool execution bug)"""
    xml = """<readfile>
./file1.pdf
</readfile><readfile>
./file2.pdf
</readfile>"""

    print("\n=== Test 8: Duplicate/sequential tags ===")
    print(f"Input: {xml}")
    try:
        blocks = extract_blocks(xml, "readfile")
        print(f"✅ SUCCESS: Extracted {len(blocks)} blocks")
        if blocks:
            for i, block in enumerate(blocks, 1):
                print(f"Block {i}: {repr(block.strip())}")
        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
        assert blocks[0].strip() == "./file1.pdf", f"Block 1 mismatch: {repr(blocks[0])}"
        assert blocks[1].strip() == "./file2.pdf", f"Block 2 mismatch: {repr(blocks[1])}"
    except Exception as e:
        print(f"❌ FAILED: {e}")


def main():
    print("="*80)
    print("XML PARSING INFRASTRUCTURE TEST SUITE")
    print("="*80)

    test_case_1_basic_visualresponse()
    test_case_2_latex_with_comparison()
    test_case_3_actual_failing_content()
    test_case_4_direct_xml_parse()
    test_case_5_escaped_xml()
    test_case_6_cdata_section()
    test_case_7_extract_blocks_behavior()
    test_case_8_duplicate_tags()

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
