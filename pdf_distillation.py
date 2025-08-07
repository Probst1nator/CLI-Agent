#!/usr/bin/env python3

import os
import argparse
import asyncio
import re
from typing import List, Optional

# Attempt to import necessary libraries, providing user-friendly errors.
try:
    from termcolor import colored
except ImportError:
    print("Package 'termcolor' not found. Please install it: pip install termcolor")
    exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Package 'python-dotenv' not found. Please install it: pip install python-dotenv")
    exit(1)
    
try:
    import PyPDF2
except ImportError:
    print(colored("PyPDF2 is not installed. Please install it to read PDF files:", "red"))
    print(colored("pip install PyPDF2", "yellow"))
    exit(1)

# Import core modules from the cli-agent project structure
try:
    from py_classes.cls_llm_router import LlmRouter
    from py_classes.cls_chat import Chat, Role
    from py_classes.enum_ai_strengths import AIStrengths
    from py_classes.globals import g
except ImportError as e:
    print(colored(f"Could not import cli-agent classes: {e}", "red"))
    print(colored("Please ensure this script is run from the root directory of the 'cli-agent' project,", "yellow"))
    print(colored("or that the project's root directory is in your PYTHONPATH.", "yellow"))
    exit(1)
    
# --- Constants ---
# Set a conservative chunk size in characters. 15,000 chars is ~3,750 tokens, leaving ample space
# in most modern LLM context windows for prompts and the model's response.
CHARACTER_CHUNK_SIZE = 15000

# --- Helper Functions ---

def setup_arguments() -> argparse.Namespace:
    """Sets up and parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="A tool to read all PDFs in a folder, summarize them individually, and then create a final meta-analysis in HTML format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-folder",
        required=True,
        help="Path to the folder containing the PDF files to analyze."
    )
    parser.add_argument(
        "-o", "--output-html",
        required=True,
        help="Path to save the final HTML meta-analysis report."
    )
    parser.add_argument(
        "--strong-model",
        default='claude-3-opus-20240229',
        help="Optional: Specify the strong model for the final meta-analysis.\nDefaults to 'claude-3-opus-20240229'."
    )
    return parser.parse_args()

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extracts all text content from a given PDF file."""
    filename = os.path.basename(pdf_path)
    print(colored(f"-> Reading PDF: {filename}", "cyan"))
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                print(colored(f"   [!] Warning: PDF '{filename}' is encrypted and cannot be read.", "yellow"))
                return None
            
            text_parts = [page.extract_text() for page in reader.pages]
            full_text = "\n".join(filter(None, text_parts))

            if not full_text.strip():
                print(colored(f"   [!] Warning: No text could be extracted from '{filename}'. It may be image-based.", "yellow"))
                return None
                
            return full_text
    except PyPDF2.errors.PdfReadError as e:
        print(colored(f"   [!] Error: Could not read PDF '{filename}'. It may be corrupted. Details: {e}", "red"))
        return None
    except Exception as e:
        print(colored(f"   [!] An unexpected error occurred while reading '{filename}': {e}", "red"))
        return None

def clean_html_response(llm_response: str) -> str:
    """Extracts the pure HTML from an LLM response that might include conversational text."""
    match = re.search(r'<!DOCTYPE html>.*</html>', llm_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0)
    
    match = re.search(r'<html.*</html>', llm_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0)

    print(colored("[!] Warning: Could not find a complete HTML document. Saving the raw LLM output.", "yellow"))
    return llm_response

# --- Core Asynchronous Functions ---

async def summarize_text_chunk(text_chunk: str, chunk_index: int, total_chunks: int) -> Optional[str]:
    """Summarizes a single chunk of text from a larger document."""
    instruction = "You are a research analyst. Your task is to summarize the following text chunk. This is one part of a larger document. Focus on extracting the key facts, findings, and arguments presented in this specific section."
    
    chat = Chat(instruction_message=instruction, debug_title=f"Chunk {chunk_index + 1}/{total_chunks}")
    chat.add_message(Role.USER, f"This is chunk {chunk_index + 1} of {total_chunks}. Please summarize it:\n\n---\n\n{text_chunk}")

    try:
        summary = await LlmRouter.generate_completion(chat, temperature=0.1)
        return summary
    except Exception as e:
        print(colored(f"   [!] LLM Error during summary of chunk {chunk_index + 1}: {e}", "red"))
        return None

async def reduce_chunk_summaries(chunk_summaries: List[str], filename: str) -> Optional[str]:
    """Combines summaries of chunks into a single, cohesive summary for the whole document."""
    print(colored(f"   -> Reducing {len(chunk_summaries)} chunk summaries for: {filename}", "blue"))
    instruction = """You are a research analyst. You have been given a series of summaries from sequential chunks of a single research paper. Your task is to synthesize these partial summaries into one final, coherent summary that represents the entire document.
    
    From this synthesized view, you must extract and structure the following key points:
    1.  **Core Insights:** What are the main findings or conclusions of the entire document?
    2.  **Progress/Contribution:** How does this work contribute to its field?
    3.  **Reliability & Foundations:** On what evidence or foundation are the conclusions based?
    
    Present your final summary in a clear, structured markdown format.
    """
    chat = Chat(instruction_message=instruction, debug_title=f"Reduce for {filename}")
    combined_chunks = "\n\n---\n\n".join(
        [f"Summary of Chunk {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)]
    )
    chat.add_message(Role.USER, f"Please synthesize the following chunk summaries from the document '{filename}':\n\n{combined_chunks}")

    try:
        final_summary = await LlmRouter.generate_completion(chat, temperature=0.1)
        return final_summary
    except Exception as e:
        print(colored(f"   [!] LLM Error during reduce step for '{filename}': {e}", "red"))
        return None

async def summarize_document_text(pdf_text: str, filename: str) -> Optional[str]:
    """Orchestrates the summarization of a document, handling chunking if necessary."""
    print(colored(f"-> Summarizing text from: {filename}...", "cyan"))
    
    if len(pdf_text) <= CHARACTER_CHUNK_SIZE:
        # Document is small enough, summarize in one go.
        print(colored("   -> Document is small. Summarizing directly.", "blue"))
        # Reuse the 'reduce' prompt as it's a good single-document summary prompt.
        return await reduce_chunk_summaries([pdf_text], filename)

    # Document is large, requires map-reduce chunking.
    print(colored(f"   -> Document is large ({len(pdf_text)} chars). Splitting into chunks.", "blue"))
    chunks = [pdf_text[i:i + CHARACTER_CHUNK_SIZE] for i in range(0, len(pdf_text), CHARACTER_CHUNK_SIZE)]
    print(colored(f"   -> Created {len(chunks)} chunks. Summarizing each...", "blue"))

    tasks = [summarize_text_chunk(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
    chunk_summaries = await asyncio.gather(*tasks)
    
    # Filter out any chunks that failed to summarize
    successful_chunk_summaries = [s for s in chunk_summaries if s]
    if not successful_chunk_summaries:
        print(colored(f"   [!] All chunks failed to summarize for '{filename}'.", "red"))
        return None

    # Reduce the successful summaries into a final one
    final_summary = await reduce_chunk_summaries(successful_chunk_summaries, filename)
    return final_summary

async def create_meta_summary(summaries: List[str], successful_sources: List[str], failed_sources: List[str], strong_model: str) -> Optional[str]:
    """Takes a list of individual summaries and creates a final, comprehensive meta-analysis in HTML."""
    print(colored("\n-> Starting final meta-analysis with a strong model...", "magenta"))
    
    instruction = """You are a distinguished academic and web designer tasked with creating a web page that presents a meta-study.
    You have been provided with summaries from various research papers, a list of successfully analyzed papers, and a list of papers that failed analysis.
    
    Your goal is to:
    1.  **Synthesize, DO NOT list:** Synthesize the information from the summaries into a coherent narrative. Identify overarching themes, points of agreement, contradictions, and the overall trajectory of the field.
    2.  **Structure as a Meta-Study:** The final output must be a single, complete HTML file. Structure it like a professional online research publication with sections like an Introduction, Key Thematic Areas, Methodological Trends, and a Synthesis/Future Outlook.
    3.  **Design the HTML:** The HTML must be well-formed and styled with clean, modern internal CSS. Use semantic tags (`<header>`, `<article>`, `<footer>`).
    4.  **Include Source Attribution:** At the end of the document, you MUST include two lists in the footer or a dedicated final section:
        - A list titled "Source Documents" containing all successfully analyzed papers.
        - A list titled "Documents That Could Not Be Processed" containing all papers that failed. If this list is empty, omit this section.
    
    Produce ONLY the full HTML content, starting with `<!DOCTYPE html>` and ending with `</html>`.
    """
    
    chat = Chat(instruction_message=instruction, debug_title="Final Meta-Analysis")
    
    # Construct the final prompt for the LLM
    content_for_llm = "Please synthesize the following document summaries into a meta-study web page.\n\n"
    content_for_llm += "# Document Summaries:\n"
    content_for_llm += "\n\n".join([f"--- SUMMARY FROM: {successful_sources[i]} ---\n\n{s}" for i, s in enumerate(summaries)])
    content_for_llm += "\n\n# Source Lists for Attribution:\n"
    content_for_llm += "## Successfully Analyzed Documents:\n- " + "\n- ".join(successful_sources)
    if failed_sources:
        content_for_llm += "\n\n## Documents That Could Not Be Processed:\n- " + "\n- ".join(failed_sources)

    chat.add_message(Role.USER, content_for_llm)
    
    try:
        html_report = await LlmRouter.generate_completion(
            chat,
            preferred_models=[strong_model],
            strengths=[AIStrengths.STRONG],
            temperature=0.3
        )
        print(colored("   [✔] Final meta-analysis generated successfully.", "green"))
        return html_report
    except Exception as e:
        print(colored(f"   [!] LLM Error during final meta-analysis: {e}", "red"))
        return None

async def main():
    """Main execution function."""
    load_dotenv(g.CLIAGENT_ENV_FILE_PATH)
    args = setup_arguments()

    if not os.path.isdir(args.input_folder):
        print(colored(f"Error: Input folder not found at '{args.input_folder}'", "red"))
        return

    print(colored("--- Starting PDF Distillation Process ---", "green", attrs=["bold"]))
    
    individual_summaries, successful_files, failed_files = [], [], []
    pdf_files = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith('.pdf')])

    if not pdf_files:
        print(colored(f"No PDF files found in '{args.input_folder}'.", "yellow"))
        return

    print(colored(f"Found {len(pdf_files)} PDF file(s) to process.", "green"))

    for filename in pdf_files:
        pdf_path = os.path.join(args.input_folder, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if pdf_text:
            summary = await summarize_document_text(pdf_text, filename)
            if summary:
                individual_summaries.append(summary)
                successful_files.append(filename)
                print(colored(f"   [✔] Successfully processed and summarized {filename}", "green"))
            else:
                failed_files.append(filename)
                print(colored(f"   [!] Failed to generate a final summary for {filename}", "red"))
            # A small delay to avoid hitting API rate limits
            await asyncio.sleep(1)
        else:
            failed_files.append(filename)
            # No summary could be generated because text extraction failed.

    if not individual_summaries:
        print(colored("\nNo summaries could be generated. Halting process.", "red"))
        if failed_files:
            print(colored("The following files could not be processed:", "yellow"))
            for f in failed_files:
                print(colored(f"- {f}", "yellow"))
        return

    print(colored(f"\nGenerated {len(individual_summaries)} individual summaries from {len(successful_files)} documents.", "green"))

    final_report = await create_meta_summary(individual_summaries, successful_files, failed_files, args.strong_model)

    if final_report:
        final_html = clean_html_response(final_report)
        try:
            with open(args.output_html, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(colored("\n--- Process Complete ---", "green", attrs=["bold"]))
            print(colored(f"Successfully saved the meta-analysis to: {args.output_html}", "green"))
        except Exception as e:
            print(colored(f"Error saving the final HTML file: {e}", "red"))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nProcess interrupted by user. Shutting down.", "yellow"))
    except Exception as e:
        print(colored(f"\nAn unexpected error occurred: {e}", "red"))