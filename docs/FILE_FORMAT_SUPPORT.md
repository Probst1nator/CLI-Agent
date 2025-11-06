# File Format Support

## Processing Paths

### PDF Files (`.pdf`)
**Detection:** Extension or magic bytes `%PDF-`
**Processing:** Multi-method extraction (PyPDF2 → pdftotext → OCR)
**Output:** Extracted text + page images for vision models

### Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.gif`, `.svg`)
**Detection:** Extension, MIME type, or magic bytes
**Processing:** Base64 encoding + registration with `tool_context`
**Output:** Image available for vision model analysis

### Plain Text (All Others)
**Formats:** `.txt`, `.md`, `.py`, `.cpp`, `.java`, `.js`, `.json`, `.csv`, `.xml`, `.html`, `.css`, `.tex`, etc.
**Processing:** UTF-8 read with Latin-1 fallback
**Output:** Raw text content

## Known Issue: Office Documents

**Formats:** `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt`, `.odt`, `.ods`, `.odp`, `.odg`
**Current Status:** ⚠️ Read as plain text → produces garbled binary output
**Reason:** ZIP-based XML formats not specially handled
**Solution:** Need dedicated extraction libraries (python-docx, openpyxl, odfpy)

## Whitelist Coverage

- **Specially handled:** 15 formats (1 PDF + 14 images)
- **Plain text:** 20 formats (code, data, markup)
- **Problematic:** 10 formats (office documents)
