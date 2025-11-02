"""
PDF processing utilities for CLI-Agent.

This module consolidates all PDF-related functionality that was previously
duplicated across multiple files (shared/common_utils.py, shared/audio/audio_utils.py).
"""

import hashlib
import os
import pickle
import re
from io import StringIO

from core.globals import g


from typing import Optional
def clean_pdf_text(text: str) -> str:
    """
    Clean and format extracted PDF text for better readability.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned and formatted text
    """
    # Step 1: Handle unicode characters (preserving special characters)
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # Step 2: Remove excessive newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # Step 3: Join split words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Step 4: Separate numbers and text
    text = re.sub(r'(\d+)([A-Za-zÄäÖöÜüß])', r'\1 \2', text)
    text = re.sub(r'([A-Za-zÄäÖöÜüß])(\d+)', r'\1 \2', text)
    # Step 5: Add space after periods if missing
    text = re.sub(r'\.(\w)', r'. \1', text)
    # Step 6: Capitalize first letter after period and newline
    text = re.sub(r'(^|\. )([a-zäöüß])', lambda m: m.group(1) + m.group(2).upper(), text)
    # Step 7: Format Euro amounts
    text = re.sub(r'(\d+)\s*Euro', r'\1 Euro', text)
    # Step 8: Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()


def get_cache_file_path(file_path: str, cache_key: str) -> str:
    """
    Generate cache file path for PDF processing results.

    Args:
        file_path: Path to the original PDF file
        cache_key: Unique identifier for the type of cached content

    Returns:
        Path to the cache file
    """
    cache_dir = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "pdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    last_modified = os.path.getmtime(file_path)
    full_cache_key = hashlib.md5(f"{file_path}_{last_modified}".encode()).hexdigest()
    return os.path.join(cache_dir, f"{cache_key}_{full_cache_key}.pickle")


def load_from_cache(cache_file: str) -> str | Optional[list[str]]:
    """
    Load cached content from pickle file.

    Args:
        cache_file: Path to cache file

    Returns:
        Cached content or None if not found
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def save_to_cache(cache_file: str, content: str | list[str]) -> None:
    """
    Save content to pickle cache file.

    Args:
        cache_file: Path to cache file
        content: Content to cache
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(content, f)


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from PDF file page by page.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of text content for each page
    """
    # Lazy import pdfminer modules
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from termcolor import colored

    resource_manager = PDFResourceManager()
    page_contents = []

    with open(pdf_path, 'rb') as fh:
        pages = list(PDFPage.get_pages(fh, caching=True, check_extractable=True))
        for i, page in enumerate(pages):
            fake_file_handle = StringIO()
            converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams(all_texts=True))
            page_interpreter = PDFPageInterpreter(resource_manager, converter)

            page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()
            page_contents.append(clean_pdf_text(text))

            converter.close()
            fake_file_handle.close()

            print(colored(f"{i+1}/{len(pages)}. Extracted page from '{pdf_path}'", "green"))

    return page_contents




def extract_pdf_content(file_path: str) -> str:
    """
    Extract all PDF content as a single string with caching.

    Args:
        file_path: Path to PDF file

    Returns:
        Complete text content of the PDF
    """
    cache_file = get_cache_file_path(file_path, "full_text")
    cached_content = load_from_cache(cache_file)

    if cached_content is not None:
        return cached_content

    page_contents = extract_text_from_pdf(file_path)
    full_content = "\n".join(page_contents)

    save_to_cache(cache_file, full_content)
    return full_content


