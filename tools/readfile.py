# tools/readfile.py
"""
This file implements the 'readfile' tool. It allows the agent to read the
content of a file from the local file system. To prevent overwhelming the
context window, it truncates large files, showing the beginning and end.
Now supports reading images and registering them as base64 data for vision models.
If the file is not found at the specified path, it will attempt a recursive
search from the expected parent directory.
"""
import base64
import hashlib
import json
import mimetypes
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from core.tool_context import tool_context
    TOOL_CONTEXT_AVAILABLE = True
except ImportError:
    TOOL_CONTEXT_AVAILABLE = False

# Removed TextResourceManager - now using file-based memory system

class readfile:
    """
    A tool to read the contents of a specified file.
    Paths are resolved relative to the current working directory.
    Large files are automatically truncated to fit within a reasonable context size.
    """
    # Define a character limit to prevent excessively large outputs.
    # This helps manage the LLM's context window and reduces token usage.
    CHUNK_LIMIT = 20000
    CONTENT_THRESHOLD = 20000  # Threshold for using resource manager

    # State file to track truncated reads
    STATE_FILE = os.path.expanduser("~/.readfile_state.json")

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'readfile'

    @staticmethod
    def _get_file_hash(filepath: str) -> str:
        """Generate a unique hash for a file based on its absolute path and modification time."""
        abs_path = os.path.abspath(filepath)
        try:
            mtime = os.path.getmtime(abs_path)
            content = f"{abs_path}:{mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except OSError:
            # If we can't get mtime, use just the path
            return hashlib.md5(abs_path.encode()).hexdigest()

    @staticmethod
    def _load_truncation_state() -> dict:
        """Load the state of previously truncated files."""
        try:
            if os.path.exists(readfile.STATE_FILE):
                with open(readfile.STATE_FILE) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    @staticmethod
    def _save_truncation_state(state: dict):
        """Save the state of truncated files."""
        try:
            with open(readfile.STATE_FILE, 'w') as f:
                json.dump(state, f)
        except OSError:
            # Silently fail if we can't save state
            pass

    @staticmethod
    def _mark_file_truncated(filepath: str):
        """Mark a file as having been truncated in a previous read."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        state[file_hash] = {
            'path': os.path.abspath(filepath),
            'truncated': True
        }
        readfile._save_truncation_state(state)

    @staticmethod
    def _was_file_truncated(filepath: str) -> bool:
        """Check if a file was previously truncated."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        return file_hash in state and state[file_hash].get('truncated', False)

    @staticmethod
    def _clear_truncation_state(filepath: str):
        """Clear the truncation state for a file after showing full content."""
        file_hash = readfile._get_file_hash(filepath)
        state = readfile._load_truncation_state()
        if file_hash in state:
            del state[file_hash]
            readfile._save_truncation_state(state)

    @staticmethod
    def _parse_line_range(line_spec: str) -> tuple:
        """Parse line specification into (start_line, end_line).

        Supports:
        - Single line: "15" -> (15, 15)
        - Range: "10-20" -> (10, 20)
        - Open range: "10-" -> (10, None)
        - From start: "-20" -> (1, 20)
        """
        line_spec = line_spec.strip()

        if '-' in line_spec:
            parts = line_spec.split('-', 1)
            start_str, end_str = parts[0].strip(), parts[1].strip()

            start_line = int(start_str) if start_str else 1
            end_line = int(end_str) if end_str else None

            return (start_line, end_line)
        else:
            # Single line
            line_num = int(line_spec)
            return (line_num, line_num)

    @staticmethod
    def _extract_lines(content: str, start_line: int, end_line: int = None) -> str:
        """Extract specific lines from file content."""
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Convert to 0-based indexing
        start_idx = max(0, start_line - 1)

        if end_line is None:
            # Read from start_line to end of file
            end_idx = total_lines
        else:
            end_idx = min(total_lines, end_line)

        if start_idx >= total_lines:
            return f"Error: Start line {start_line} exceeds file length ({total_lines} lines)"

        selected_lines = lines[start_idx:end_idx]

        # Add line numbers for clarity
        result_lines = []
        for i, line in enumerate(selected_lines, start=start_line):
            result_lines.append(f"{i:4d}→{line}" if line.endswith('\n') else f"{i:4d}→{line}\n")

        header = f"Lines {start_line}-{end_line if end_line else total_lines} ({total_lines} total lines):\n"
        return header + ''.join(result_lines)

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "readfile",
            "description": (
                "Supports text, PDF, image, and office document files. "
                "PDFs are automatically extracted to text (with OCR fallback). "
                "Images and PDF pages are registered for vision model analysis. "
                "Office documents (.docx, .xlsx, .pptx, .xls, .odt, .ods, .odp) are extracted to readable text. "
                "Large text files are automatically saved to memory files that you can access with <readfile>."
            ),
            "example": (
                "<readfile>./src/main.py</readfile>\n"
                "<readfile><filepath>./app.js</filepath><lines>50-75</lines></readfile>\n"
                "<readfile>./screenshot.png</readfile>\n"
                "<readfile>./document.pdf</readfile>\n"
                "<readfile>./report.docx</readfile>\n"
                "<readfile>./data.xlsx</readfile>\n"
            )
        }

    @staticmethod
    def is_pdf_file(filepath: str) -> bool:
        """Check if the file is a PDF based on its extension and magic bytes."""
        _, ext = os.path.splitext(filepath.lower())
        if ext == '.pdf':
            return True

        # Check magic bytes
        try:
            with open(filepath, 'rb') as f:
                header = f.read(5)
            return header.startswith(b'%PDF-')
        except Exception:
            return False

    @staticmethod
    def process_pdf_file(filepath: str) -> str:
        """
        Process a PDF file by extracting text and images.

        Tries multiple methods in order:
        1. PyPDF2 for text extraction
        2. pdftotext command-line tool
        3. OCR with pytesseract as fallback

        Also extracts embedded images and registers them.

        Args:
            filepath: Path to the PDF file

        Returns:
            Extracted text content with metadata
        """
        import subprocess

        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath) / 1024  # KB

        extracted_text = None
        method_used = None
        images_extracted = 0

        # Method 1: Try PyPDF2
        try:
            import PyPDF2
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                text_parts = []

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"\n{'='*60}\nPage {page_num}/{num_pages}\n{'='*60}\n{page_text}")

                if text_parts:
                    extracted_text = '\n'.join(text_parts)
                    method_used = "PyPDF2"
        except ImportError:
            pass
        except Exception:
            # PyPDF2 failed, will try other methods
            pass

        # Method 2: Try pdftotext command-line tool
        if not extracted_text:
            try:
                result = subprocess.run(
                    ['pdftotext', '-layout', filepath, '-'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    extracted_text = result.stdout
                    method_used = "pdftotext"
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass

        # Method 3: OCR fallback with pdf2image + pytesseract
        if not extracted_text:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                from PIL import Image

                # Convert PDF to images
                images = convert_from_path(filepath, dpi=200, fmt='png')
                text_parts = []

                for i, image in enumerate(images, 1):
                    # Extract text via OCR
                    page_text = pytesseract.image_to_string(image, lang='deu+eng')
                    if page_text.strip():
                        text_parts.append(f"\n{'='*60}\nPage {i}/{len(images)} (OCR)\n{'='*60}\n{page_text}")

                    # Register image with tool context for vision model
                    if TOOL_CONTEXT_AVAILABLE and i <= 5:  # Limit to first 5 pages
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_data = img_byte_arr.getvalue()
                        base64_data = base64.b64encode(img_data).decode('utf-8')
                        tool_context.add_base64_image(base64_data, f"{filename} - Page {i}")
                        images_extracted += 1

                if text_parts:
                    extracted_text = '\n'.join(text_parts)
                    method_used = "OCR (pytesseract)"

            except ImportError:
                pass
            except Exception:
                pass

        # If all methods failed
        if not extracted_text:
            return (f"📄 PDF file detected: {filename} ({file_size:.1f} KB)\n\n"
                   f"⚠️  Could not extract text from PDF. Missing dependencies:\n"
                   f"  - Install PyPDF2: pip install PyPDF2\n"
                   f"  - Install pdftotext: apt-get install poppler-utils\n"
                   f"  - Install OCR tools: pip install pdf2image pytesseract && apt-get install tesseract-ocr tesseract-ocr-deu\n\n"
                   f"Try using the bash tool to convert:\n"
                   f"<bash>pdftotext \"{filename}\" output.txt && cat output.txt</bash>")

        # Success - return extracted content
        header = f"📄 PDF: {filename} ({file_size:.1f} KB) - Extracted via {method_used}"
        if images_extracted > 0:
            header += f"\n📸 {images_extracted} page images extracted for vision analysis"

        return f"{header}\n\n{extracted_text}"

    @staticmethod
    def is_image_file(filepath: str) -> bool:
        """Check if the file is an image based on its extension, MIME type, and magic bytes."""
        # Common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tiff', '.tif'}

        # Check file extension
        _, ext = os.path.splitext(filepath.lower())
        if ext in image_extensions:
            return True

        # Also check MIME type
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type and mime_type.startswith('image/'):
            return True

        # For files without extensions or unclear MIME types, check magic bytes
        try:
            with open(filepath, 'rb') as f:
                header = f.read(20)  # Read first 20 bytes

            # Check for common image magic bytes
            if header.startswith(b'\x89PNG') or header.startswith(b'\xff\xd8\xff') or (header.startswith(b'GIF8') or header.startswith(b'BM')) or (header.startswith(b'RIFF') and b'WEBP' in header[:20] or (header.startswith(b'\x00\x00\x01\x00') or header.startswith(b'\x00\x00\x02\x00')) or header.startswith((b'II*\x00', b'MM\x00*'))):  # PNG
                return True

        except Exception:
            # If we can't read the file, fall back to extension/MIME type only
            pass

        return False

    @staticmethod
    def process_image_file(filepath: str) -> str:
        """
        Process an image file by encoding it as base64 and registering it with the tool context.

        Args:
            filepath: Path to the image file

        Returns:
            A description message about the processed image
        """
        try:
            # Get file info
            file_size = os.path.getsize(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)

            # Read and encode the image
            with open(filepath, 'rb') as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')

            # If MIME type is unknown, try to detect from magic bytes
            if not mime_type:
                header = image_data[:20] if len(image_data) >= 20 else image_data
                if header.startswith(b'\x89PNG'):
                    mime_type = 'image/png'
                elif header.startswith(b'\xff\xd8\xff'):
                    mime_type = 'image/jpeg'
                elif header.startswith(b'GIF8'):
                    mime_type = 'image/gif'
                elif header.startswith(b'BM'):
                    mime_type = 'image/bmp'
                elif header.startswith(b'RIFF') and b'WEBP' in header:
                    mime_type = 'image/webp'

            # Create description
            filename = os.path.basename(filepath)
            size_kb = file_size / 1024
            description = f"Image: {filename} ({mime_type or 'unknown type'}, {size_kb:.1f} KB)"

            # Register with tool context if available
            if TOOL_CONTEXT_AVAILABLE:
                tool_context.add_base64_image(base64_data, description)
                return f"📸 Image loaded: {description}\n[Image has been processed and is now available for analysis]"
            else:
                return f"📸 Image detected: {description}\n[Warning: Tool context not available - image analysis may be limited]"

        except Exception as e:
            return f"Error processing image file '{filepath}': {str(e)}"

    @staticmethod
    def is_office_document(filepath: str) -> tuple:
        """
        Check if the file is an office document and return (is_office, format_type).

        Returns:
            tuple: (bool, str) - (is office document, format type)
            format_type can be: 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'odt', 'ods', 'odp', 'odg'
        """
        _, ext = os.path.splitext(filepath.lower())

        # Map extensions to format types
        office_extensions = {
            '.docx': 'docx', '.doc': 'doc',
            '.xlsx': 'xlsx', '.xls': 'xls',
            '.pptx': 'pptx', '.ppt': 'ppt',
            '.odt': 'odt', '.ods': 'ods', '.odp': 'odp', '.odg': 'odg'
        }

        if ext in office_extensions:
            return (True, office_extensions[ext])

        # Check magic bytes for files without proper extensions
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)

            # ZIP-based formats (DOCX, XLSX, PPTX, ODT, ODS, ODP, ODG) start with PK
            if header.startswith(b'PK\x03\x04'):
                # Could be any ZIP-based office format
                # Try to read the content_types or mimetype file to determine
                import zipfile
                try:
                    with zipfile.ZipFile(filepath, 'r') as zf:
                        # Check for Office Open XML (Microsoft)
                        if '[Content_Types].xml' in zf.namelist():
                            content_types = zf.read('[Content_Types].xml').decode('utf-8', errors='ignore')
                            if 'wordprocessingml' in content_types:
                                return (True, 'docx')
                            elif 'spreadsheetml' in content_types:
                                return (True, 'xlsx')
                            elif 'presentationml' in content_types:
                                return (True, 'pptx')

                        # Check for OpenDocument (LibreOffice/OpenOffice)
                        if 'mimetype' in zf.namelist():
                            mimetype = zf.read('mimetype').decode('utf-8', errors='ignore').strip()
                            if 'opendocument.text' in mimetype:
                                return (True, 'odt')
                            elif 'opendocument.spreadsheet' in mimetype:
                                return (True, 'ods')
                            elif 'opendocument.presentation' in mimetype:
                                return (True, 'odp')
                            elif 'opendocument.graphics' in mimetype:
                                return (True, 'odg')
                except Exception:
                    pass

            # Old Office formats (DOC, XLS, PPT) use OLE2 format
            elif header.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
                # This is an OLE2 file, likely old Office format
                # Without deeper inspection, we can't be 100% sure which type
                # Return generic 'doc' and let the processor handle it
                return (True, 'doc')

        except Exception:
            pass

        return (False, None)

    @staticmethod
    def process_docx_file(filepath: str) -> str:
        """Extract text from a .docx file."""
        try:
            import docx

            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB

            doc = docx.Document(filepath)

            # Extract text from paragraphs
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        text_parts.append(row_text)

            if not text_parts:
                return f"📄 Word document: {filename} ({file_size:.1f} KB)\n\n[Document appears to be empty or contains no extractable text]"

            extracted_text = '\n'.join(text_parts)
            header = f"📄 Word document: {filename} ({file_size:.1f} KB)"

            return f"{header}\n\n{extracted_text}"

        except ImportError:
            return f"⚠️  Cannot read .docx file: python-docx not installed\nInstall with: pip install python-docx"
        except Exception as e:
            return f"Error reading .docx file '{filepath}': {str(e)}"

    @staticmethod
    def process_xlsx_file(filepath: str) -> str:
        """Extract text from a .xlsx file."""
        try:
            import openpyxl

            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB

            wb = openpyxl.load_workbook(filepath, data_only=True)
            text_parts = []

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]

                # Add sheet header
                text_parts.append(f"\n{'='*60}\nSheet: {sheet_name}\n{'='*60}")

                # Extract cell values
                for row in sheet.iter_rows(values_only=True):
                    # Filter out empty rows
                    row_values = [str(cell) if cell is not None else '' for cell in row]
                    if any(val.strip() for val in row_values):
                        text_parts.append(' | '.join(row_values))

            if not text_parts:
                return f"📊 Excel spreadsheet: {filename} ({file_size:.1f} KB)\n\n[Spreadsheet appears to be empty]"

            extracted_text = '\n'.join(text_parts)
            header = f"📊 Excel spreadsheet: {filename} ({file_size:.1f} KB) - {len(wb.sheetnames)} sheet(s)"

            return f"{header}\n\n{extracted_text}"

        except ImportError:
            return f"⚠️  Cannot read .xlsx file: openpyxl not installed\nInstall with: pip install openpyxl"
        except Exception as e:
            return f"Error reading .xlsx file '{filepath}': {str(e)}"

    @staticmethod
    def process_xls_file(filepath: str) -> str:
        """Extract text from a .xls file (older Excel format)."""
        try:
            import xlrd

            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB

            wb = xlrd.open_workbook(filepath)
            text_parts = []

            for sheet in wb.sheets():
                # Add sheet header
                text_parts.append(f"\n{'='*60}\nSheet: {sheet.name}\n{'='*60}")

                # Extract cell values
                for row_idx in range(sheet.nrows):
                    row_values = [str(sheet.cell_value(row_idx, col_idx)) for col_idx in range(sheet.ncols)]
                    if any(val.strip() for val in row_values):
                        text_parts.append(' | '.join(row_values))

            if not text_parts:
                return f"📊 Excel spreadsheet (legacy): {filename} ({file_size:.1f} KB)\n\n[Spreadsheet appears to be empty]"

            extracted_text = '\n'.join(text_parts)
            header = f"📊 Excel spreadsheet (legacy): {filename} ({file_size:.1f} KB) - {wb.nsheets} sheet(s)"

            return f"{header}\n\n{extracted_text}"

        except ImportError:
            return f"⚠️  Cannot read .xls file: xlrd not installed\nInstall with: pip install xlrd"
        except Exception as e:
            return f"Error reading .xls file '{filepath}': {str(e)}"

    @staticmethod
    def process_pptx_file(filepath: str) -> str:
        """Extract text from a .pptx file."""
        try:
            import pptx

            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB

            prs = pptx.Presentation(filepath)
            text_parts = []

            for slide_num, slide in enumerate(prs.slides, 1):
                # Add slide header
                text_parts.append(f"\n{'='*60}\nSlide {slide_num}\n{'='*60}")

                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text_parts.append(shape.text)

                    # Extract text from tables if present
                    if hasattr(shape, "table"):
                        for row in shape.table.rows:
                            row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                            if row_text.strip():
                                text_parts.append(row_text)

            if not text_parts:
                return f"📊 PowerPoint presentation: {filename} ({file_size:.1f} KB)\n\n[Presentation appears to be empty or contains no extractable text]"

            extracted_text = '\n'.join(text_parts)
            header = f"📊 PowerPoint presentation: {filename} ({file_size:.1f} KB) - {len(prs.slides)} slide(s)"

            return f"{header}\n\n{extracted_text}"

        except ImportError:
            return f"⚠️  Cannot read .pptx file: python-pptx not installed\nInstall with: pip install python-pptx"
        except Exception as e:
            return f"Error reading .pptx file '{filepath}': {str(e)}"

    @staticmethod
    def process_opendocument_file(filepath: str, format_type: str) -> str:
        """Extract text from OpenDocument files (.odt, .ods, .odp, .odg)."""
        try:
            from odf import text, teletype
            from odf.opendocument import load

            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB

            doc = load(filepath)
            text_parts = []

            # Extract text elements
            for element in doc.getElementsByType(text.P):
                paragraph_text = teletype.extractText(element)
                if paragraph_text.strip():
                    text_parts.append(paragraph_text)

            # Extract text from lists
            for element in doc.getElementsByType(text.List):
                list_text = teletype.extractText(element)
                if list_text.strip():
                    text_parts.append(list_text)

            # Extract text from headers
            for element in doc.getElementsByType(text.H):
                header_text = teletype.extractText(element)
                if header_text.strip():
                    text_parts.append(header_text)

            if not text_parts:
                return f"📄 OpenDocument file: {filename} ({file_size:.1f} KB)\n\n[Document appears to be empty or contains no extractable text]"

            extracted_text = '\n'.join(text_parts)

            doc_type = {
                'odt': 'OpenDocument Text',
                'ods': 'OpenDocument Spreadsheet',
                'odp': 'OpenDocument Presentation',
                'odg': 'OpenDocument Graphics'
            }.get(format_type, 'OpenDocument')

            header = f"📄 {doc_type}: {filename} ({file_size:.1f} KB)"

            return f"{header}\n\n{extracted_text}"

        except ImportError:
            return (f"⚠️  Cannot read OpenDocument file: odfpy not installed\n"
                   f"Install with: pip install odfpy\n"
                   f"Note: If installation fails, try converting to PDF or DOCX format")
        except Exception as e:
            return f"Error reading OpenDocument file '{filepath}': {str(e)}"

    @staticmethod
    def process_office_document(filepath: str, format_type: str) -> str:
        """
        Route office document to the appropriate processor based on format type.

        Args:
            filepath: Path to the office document
            format_type: Document format type (docx, xlsx, pptx, etc.)

        Returns:
            Extracted text content
        """
        if format_type == 'docx':
            return readfile.process_docx_file(filepath)
        elif format_type == 'xlsx':
            return readfile.process_xlsx_file(filepath)
        elif format_type == 'xls':
            return readfile.process_xls_file(filepath)
        elif format_type == 'pptx':
            return readfile.process_pptx_file(filepath)
        elif format_type in ('odt', 'ods', 'odp', 'odg'):
            return readfile.process_opendocument_file(filepath, format_type)
        elif format_type in ('doc', 'ppt'):
            # Old Office formats - try LibreOffice conversion as fallback
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB
            return (f"📄 Legacy Office document: {filename} ({file_size:.1f} KB)\n\n"
                   f"⚠️  Legacy .doc/.ppt formats require conversion.\n"
                   f"Recommended approaches:\n"
                   f"1. Convert with LibreOffice:\n"
                   f"   <bash>libreoffice --headless --convert-to pdf '{filepath}'</bash>\n"
                   f"2. Open in Microsoft Word/PowerPoint and save as .docx/.pptx\n"
                   f"3. Use an online converter")
        else:
            return f"⚠️  Unsupported office document format: {format_type}"

    @staticmethod
    def run(content: str) -> str:
        """
        Reads the content of the file specified in the input string.
        Supports both simple filepath format and XML format with line ranges.
        If file is not found, performs a recursive search from the parent directory.

        Args:
            content: A string containing the path to the file or XML with filepath and lines.

        Returns:
            The content of the file as a string, or an error message if the file
            cannot be read. Returns a truncated version for large files.
        """
        try:
            import re

            # Check if input is XML format with filepath and lines tags
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            lines_match = re.search(r'<lines>(.*?)</lines>', content, re.DOTALL)

            if filepath_match:
                # XML format
                filepath = filepath_match.group(1).strip()
                line_spec = lines_match.group(1).strip() if lines_match else None
            else:
                # Simple format - just filepath
                filepath = content.strip()
                line_spec = None

            if not filepath:
                return "Error: No filepath provided."

            # Resolve the path relative to the current working directory for consistency and security
            full_path = os.path.abspath(filepath)
            search_message = ""

            # Check if the file exists. If not, perform a fallback recursive search.
            if not os.path.isfile(full_path):
                filename_to_find = os.path.basename(filepath)
                search_root = os.path.dirname(full_path)

                if not os.path.isdir(search_root):
                     return f"Error: File not found at '{full_path}' and parent directory '{search_root}' does not exist to start a search."

                found_files = []
                for root, dirs, files in os.walk(search_root):
                    # Prune common large/binary directories to speed up search
                    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.venv', 'build', 'dist']]
                    if filename_to_find in files:
                        found_files.append(os.path.join(root, filename_to_find))

                if len(found_files) == 1:
                    original_full_path = full_path
                    full_path = found_files[0]  # Update the path to the found file
                    search_message = f"File not found at '{original_full_path}'. Found at '{full_path}' instead. Reading from new location.\n\n"
                elif len(found_files) > 1:
                    paths_str = '\n - '.join(found_files)
                    return f"Error: File '{filename_to_find}' not found at specified path. Found multiple possible matches in '{search_root}':\n - {paths_str}\nPlease provide a more specific path."
                else:
                     return f"Error: File not found at '{full_path}' (and not found in a recursive search from '{search_root}')."

            # Check if this is a PDF file
            if readfile.is_pdf_file(full_path):
                pdf_result = readfile.process_pdf_file(full_path)
                return search_message + pdf_result

            # Check if this is an image file
            if readfile.is_image_file(full_path):
                image_result = readfile.process_image_file(full_path)
                return search_message + image_result

            # Check if this is an office document
            is_office, format_type = readfile.is_office_document(full_path)
            if is_office:
                office_result = readfile.process_office_document(full_path, format_type)
                return search_message + office_result

            # Attempt to read the file with UTF-8 encoding, replacing errors
            try:
                with open(full_path, encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
            except Exception:
                # Fallback for non-UTF-8 files
                try:
                    with open(full_path, encoding='latin-1') as f:
                        file_content = f.read()
                except Exception as read_error:
                     return f"Error: Could not read file '{full_path}'. Tried UTF-8 and Latin-1. Reason: {read_error}"

            # Handle line-indexed reading if line_spec is provided
            if line_spec:
                try:
                    start_line, end_line = readfile._parse_line_range(line_spec)
                    line_content = readfile._extract_lines(file_content, start_line, end_line)
                    return search_message + line_content
                except ValueError:
                    return f"Error: Invalid line specification '{line_spec}'. Use formats like '15', '10-20', '10-', or '-20'."
                except Exception as e:
                    return f"Error extracting lines: {str(e)}"

            # Check if this file was previously truncated and user wants full output
            was_previously_truncated = readfile._was_file_truncated(full_path)

            # Clear any previous truncation state since general solution handles large files
            if was_previously_truncated:
                readfile._clear_truncation_state(full_path)

            # Track recently accessed file path for autocomplete
            try:
                from core.globals import g
                if full_path not in g.RECENT_FILE_PATHS:
                    g.RECENT_FILE_PATHS.insert(0, full_path)
                    g.RECENT_FILE_PATHS = g.RECENT_FILE_PATHS[:g.MAX_RECENT_FILES]
            except Exception:
                # Silently fail if globals not available
                pass

            # Return full content - main.py will automatically create resource if too large
            return search_message + file_content

        except Exception as e:
            return f"An unexpected error occurred in the readfile tool: {str(e)}"