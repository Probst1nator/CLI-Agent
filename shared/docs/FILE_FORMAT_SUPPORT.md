# File Format Support

The `readfile` tool provides comprehensive support for reading and extracting content from a wide variety of file formats. All formats are processed securely with read-only operations.

## Supported Formats

### 📄 PDF Files (`.pdf`)
**Detection:** Extension or magic bytes `%PDF-`
**Processing:** Multi-method extraction (PyPDF2 → pdftotext → OCR)
**Output:** Extracted text with page separators + page images for vision models
**Dependencies:** `PyPDF2` (optional: `pdftotext`, `pytesseract`, `pdf2image`)

**Example:**
```
📄 PDF: document.pdf (1.2 MB) - Extracted via PyPDF2
📸 5 page images extracted for vision analysis

============================================================
Page 1/10
============================================================
[Extracted text from page 1...]
```

---

### 🖼️ Images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.svg`, `.ico`, `.tiff`, `.tif`)
**Detection:** Extension, MIME type, or magic bytes
**Processing:** Base64 encoding + registration with `tool_context`
**Output:** Image available for vision model analysis
**Dependencies:** Built-in (no external libraries required)

**Example:**
```
📸 Image loaded: screenshot.png (image/png, 342.5 KB)
[Image has been processed and is now available for analysis]
```

---

### 📊 Office Documents

#### Microsoft Word (`.docx`)
**Detection:** Extension or ZIP magic bytes + content type check
**Processing:** Text and table extraction from paragraphs and tables
**Output:** Formatted text with tables
**Dependencies:** `python-docx`

**Example:**
```
📄 Word document: report.docx (36.0 KB)

Project Report
This quarter's sales exceeded expectations...

Header 1 | Header 2 | Header 3
Data 1   | Data 2   | Data 3
```

#### Microsoft Excel (`.xlsx`, `.xls`)
**Detection:** Extension or ZIP/OLE2 magic bytes
**Processing:** All sheets with cell data extraction
**Output:** Multi-sheet data with headers
**Dependencies:** `openpyxl` (`.xlsx`), `xlrd` (`.xls`)

**Example:**
```
📊 Excel spreadsheet: data.xlsx (5.3 KB) - 2 sheet(s)

============================================================
Sheet: Sales Data
============================================================
Month    | Revenue | Expenses
January  | 50000   | 30000
February | 55000   | 32000
```

#### Microsoft PowerPoint (`.pptx`)
**Detection:** Extension or ZIP magic bytes + content type check
**Processing:** Text extraction from all slides and tables
**Output:** Slide-by-slide content
**Dependencies:** `python-pptx`

**Example:**
```
📊 PowerPoint presentation: slides.pptx (29.3 KB) - 5 slide(s)

============================================================
Slide 1
============================================================
Quarterly Review
Q4 2024 Results
```

#### OpenDocument Formats (`.odt`, `.ods`, `.odp`, `.odg`)
**Detection:** Extension or ZIP magic bytes + mimetype check
**Processing:** Text extraction from paragraphs, lists, and headers
**Output:** Formatted document text
**Dependencies:** `odfpy` (optional, with graceful fallback)

**Example:**
```
📄 OpenDocument Text: document.odt (12.1 KB)

Document Title
Body text with formatting...
```

#### Legacy Office Formats (`.doc`, `.ppt`)
**Detection:** OLE2 magic bytes `\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1`
**Processing:** Provides conversion guidance
**Output:** Instructions for LibreOffice conversion or format upgrade
**Dependencies:** None (guidance only)

---

### 📊 CSV/TSV Files (`.csv`, `.tsv`)
**Detection:** Extension
**Processing:** Automatic delimiter detection + table formatting
**Output:** Aligned columns with header separation
**Dependencies:** Built-in `csv` module

**Features:**
- Auto-detects delimiters (comma, tab, or CSV sniffer)
- Column alignment with visual separators
- Shows row and column counts
- Limits to first 100 rows for large files

**Example:**
```
📊 CSV file: data.csv (12.4 KB) - 245 rows × 4 columns
(Showing first 100 rows)

Name           | Age | City        | Occupation
---------------+-----+-------------+-----------
Alice Smith    | 25  | New York    | Engineer
Bob Johnson    | 30  | Los Angeles | Designer
```

---

### 🌐 HTML Files (`.html`, `.htm`)
**Detection:** Extension or `<!DOCTYPE html>` / `<html>` in content
**Processing:** Text extraction with tag filtering
**Output:** Readable text content without markup
**Dependencies:** Built-in `html.parser`

**Features:**
- Strips `<script>`, `<style>`, `<head>`, `<meta>`, `<link>` tags
- Preserves document structure
- Extracts text from paragraphs, headings, lists, etc.

**Example:**
```
🌐 HTML file: page.html (4.2 KB)

Welcome to Our Site
This is the main content...
Section 1
Details about section 1...
```

---

### 📦 Archive Files

#### ZIP Archives (`.zip`)
**Detection:** Extension or magic bytes `PK\x03\x04`
**Processing:** Secure file listing (no extraction)
**Output:** File tree with sizes
**Dependencies:** Built-in `zipfile` module

**Example:**
```
📦 Archive: backup.zip (45.2 KB, uncompressed: 156.8 KB)
12 file(s):

  src/main.py (3.2 KB)
  src/utils.py (1.8 KB)
  data/config.json (0.5 KB)
  tests/test_main.py (2.1 KB)
  ... and 8 more files
```

#### TAR Archives (`.tar`, `.tar.gz`, `.tar.bz2`, `.tgz`)
**Detection:** Extension or tarfile validation
**Processing:** Secure file listing with compression awareness
**Output:** File tree with sizes
**Dependencies:** Built-in `tarfile` module

**Example:**
```
📦 Archive: release.tar.gz (128.4 KB, uncompressed: 512.0 KB)
34 file(s):

  project/src/core.py (12.3 KB)
  project/src/utils.py (8.5 KB)
  project/README.md (4.2 KB)
  ...
```

#### GZIP Files (`.gz`)
**Detection:** Magic bytes `\x1f\x8b`
**Processing:** Single file compression info
**Output:** Compressed file information
**Dependencies:** Built-in `gzip` module

---

### 🗄️ SQLite Databases (`.db`, `.sqlite`, `.sqlite3`, `.db3`)
**Detection:** Extension or magic bytes `SQLite format 3\x00`
**Processing:** **Read-only** schema inspection and data sampling
**Output:** Tables, columns, types, constraints, and sample data
**Dependencies:** Built-in `sqlite3` module

**Features:**
- Read-only mode (100% safe, no modifications possible)
- Shows all tables with row counts
- Complete schema with column types and constraints
- First 5 rows of sample data per table
- Identifies primary keys and data types

**Example:**
```
🗄️  SQLite database: app.db (142.5 KB)
3 table(s):

============================================================
Table: users (1,247 rows)
============================================================
Columns:
  - id: INTEGER (PRIMARY KEY)
  - username: TEXT
  - email: TEXT
  - created_at: TIMESTAMP

Sample data (first 5 rows):
  id | username | email               | created_at
  ---+----------+---------------------+-------------------
  1  | alice    | alice@example.com   | 2024-01-15 10:23:45
  2  | bob      | bob@example.com     | 2024-01-16 14:30:12
  ...
```

---

### 📓 Jupyter Notebooks (`.ipynb`)
**Detection:** Extension `.ipynb`
**Processing:** JSON parsing with cell formatting
**Output:** Formatted cells with code, markdown, and outputs
**Dependencies:** Built-in `json` module

**Features:**
- Shows both code and markdown cells
- Includes cell outputs (stdout, results, display data)
- Cell type indicators
- Limits to first 3 outputs per cell

**Example:**
```
📓 Jupyter Notebook: analysis.ipynb (24.3 KB)
15 cell(s):

============================================================
Cell 1 [markdown]
============================================================
# Data Analysis
This notebook analyzes customer behavior...

============================================================
Cell 2 [code]
============================================================
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('customers.csv')

Output:
<pandas.DataFrame with 1000 rows × 8 columns>

============================================================
Cell 3 [code]
============================================================
df.describe()

Output:
       age         purchases
count  1000.00     1000.00
mean   34.52       145.33
...
```

---

### 📝 Plain Text Files

**Formats:** `.txt`, `.md`, `.py`, `.cpp`, `.java`, `.js`, `.json`, `.xml`, `.css`, `.tex`, `.sh`, `.yaml`, `.yml`, `.toml`, `.ini`, `.conf`, `.log`, and many more

**Processing:** UTF-8 read with Latin-1 fallback
**Output:** Raw text content with optional line-range support
**Dependencies:** None (built-in)

**Features:**
- Line-range support: `<lines>10-20</lines>`, `<lines>50-</lines>`, `<lines>-30</lines>`
- Automatic truncation for very large files
- Preserves formatting and whitespace

**Example:**
```
<readfile><filepath>./src/main.py</filepath><lines>50-75</lines></readfile>

Lines 50-75 (250 total lines):
  50→def process_data(input_file):
  51→    """Process input data and return results."""
  52→    with open(input_file, 'r') as f:
  ...
```

---

## Format Coverage Summary

| Category | Count | Formats |
|----------|-------|---------|
| **Office Documents** | 10 | `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt`, `.odt`, `.ods`, `.odp`, `.odg` |
| **Data Files** | 3 | `.csv`, `.tsv`, `.db`/`.sqlite` |
| **Archives** | 7 | `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tgz`, `.gz`, `.bz2` |
| **Documents** | 3 | `.pdf`, `.html`, `.htm` |
| **Notebooks** | 1 | `.ipynb` |
| **Images** | 10+ | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.svg`, `.ico`, `.tiff`, `.tif` |
| **Plain Text** | 50+ | `.txt`, `.md`, `.py`, `.js`, `.json`, `.xml`, `.yaml`, etc. |

**Total:** 80+ file format variations supported

---

## Security & Performance

### Security Features
- **Read-only operations** - No file modifications possible
- **No code execution** - All processing is data extraction only
- **Safe archive handling** - Lists contents without extraction
- **SQLite read-only mode** - Prevents any database modifications
- **Magic byte validation** - Detects file types beyond extensions

### Performance Optimizations
- **Smart truncation** - Limits output for large files (100 rows for CSV, 100 files for archives)
- **Lazy loading** - Only imports libraries when needed
- **Memory efficient** - Streams large files when possible
- **Context management** - Automatic truncation for LLM context limits

### Error Handling
- **Graceful fallbacks** - Multiple extraction methods for PDFs
- **Clear error messages** - Installation instructions when libraries missing
- **Format detection** - Works even without file extensions
- **Encoding resilience** - UTF-8 with Latin-1 fallback for text files

---

## Dependencies

### Required (Already Installed)
- `PyMuPDF` - PDF processing
- `python-docx` - Word documents
- `python-pptx` - PowerPoint presentations
- `openpyxl` - Excel (.xlsx)
- `xlrd` - Excel (.xls)

### Optional (Graceful Degradation)
- `odfpy` - OpenDocument formats (shows conversion instructions if missing)
- `pdftotext` - Alternative PDF extraction
- `pytesseract` + `pdf2image` - PDF OCR fallback

### Built-in (No Installation Needed)
- `csv` - CSV/TSV parsing
- `html.parser` - HTML text extraction
- `zipfile` - ZIP archives
- `tarfile` - TAR archives
- `gzip` - GZIP files
- `sqlite3` - SQLite databases
- `json` - Jupyter notebooks

---

## Usage Examples

### Reading Different Formats
```xml
<!-- Office documents -->
<readfile>./report.docx</readfile>
<readfile>./data.xlsx</readfile>
<readfile>./slides.pptx</readfile>

<!-- Data files -->
<readfile>./customers.csv</readfile>
<readfile>./app.db</readfile>

<!-- Archives (safe inspection) -->
<readfile>./backup.zip</readfile>
<readfile>./release.tar.gz</readfile>

<!-- Documents -->
<readfile>./document.pdf</readfile>
<readfile>./page.html</readfile>

<!-- Notebooks -->
<readfile>./analysis.ipynb</readfile>

<!-- Images (for vision models) -->
<readfile>./screenshot.png</readfile>

<!-- Text with line ranges -->
<readfile><filepath>./src/main.py</filepath><lines>50-100</lines></readfile>
```

---

## Format Detection

The tool uses a multi-step detection process:

1. **File extension** - Primary detection method
2. **Magic bytes** - Validates file type for extensionless files
3. **Content inspection** - ZIP archives checked for Office/OpenDocument formats
4. **Fallback handling** - Graceful degradation when formats unclear

**Examples of magic byte detection:**
- ZIP files: `PK\x03\x04`
- PDF files: `%PDF-`
- SQLite: `SQLite format 3\x00`
- GZIP: `\x1f\x8b`
- PNG: `\x89PNG`
- JPEG: `\xff\xd8\xff`

This ensures files are processed correctly even without proper extensions.

---

## Known Limitations

1. **Office documents without libraries** - Requires python-docx, openpyxl, python-pptx
2. **Very large archives** - Limited to first 100 files listed
3. **Binary formats** - Not supported (executables, compiled code, proprietary formats)
4. **Encrypted files** - Cannot read password-protected documents
5. **Complex Excel formulas** - Shows values only (formulas not evaluated)

---

## Future Enhancements

Potential additions based on user needs:
- Email formats (`.eml`, `.msg`)
- E-book formats (`.epub`, `.mobi`)
- RTF documents
- Parquet/Feather data files
- Media metadata extraction
- Certificate inspection (`.pem`, `.crt`)
