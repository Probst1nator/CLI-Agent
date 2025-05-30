import json
import os
import os
import re
import pickle
import hashlib
from io import StringIO
from typing import Dict, Iterable, List, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime
from termcolor import colored

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g

# Only for type checking, not actually importing at runtime
if TYPE_CHECKING:
    import chromadb


class RagTooling:
    @classmethod
    def pdf_or_folder_to_database(cls, pdf_or_folder_path: str, collection: 'chromadb.Collection', topology_model_key: str = "phi3.5", force_local: bool = True) -> 'chromadb.Collection':
        """
        Extracts content from a PDF file or multiple PDFs in a folder (and its subfolders),
        processes them into propositions, and stores them in a Chroma database.
        This function performs the following steps for each PDF:
        1. Extracts text and image content from the PDF.
        2. Splits the text content into digestible chunks.
        3. Converts each chunk into propositions.
        4. Embeds and stores each proposition in the database.
        Args:
        pdf_or_folder_path (str): The file path of a single PDF or a folder containing multiple PDFs.
        collection (chromadb.Collection): The collection to store the extracted propositions in.
        Raises:
        FileNotFoundError: If the pdf_or_folder_path does not exist.
        ValueError: If the pdf_or_folder_path is neither a file nor a directory.
        """
        if not os.path.exists(pdf_or_folder_path):
            raise FileNotFoundError(f"The path {pdf_or_folder_path} does not exist.")

        if os.path.isfile(pdf_or_folder_path) and pdf_or_folder_path.lower().endswith('.pdf'):
            # Process a single PDF file
            cls._process_single_pdf(pdf_or_folder_path, collection, topology_model_key=topology_model_key, force_local=force_local)
        elif os.path.isdir(pdf_or_folder_path):
            pdf_files = [os.path.join(root, file) for root, _, files in os.walk(pdf_or_folder_path) 
                        for file in files if file.lower().endswith('.pdf')]
            total_files = len(pdf_files)
            
            for index, file_path in enumerate(pdf_files, start=1):
                message = f"({index}/{total_files}) Processing file: {file_path}"
                print(colored(message, 'green'))
                cls._process_single_pdf(file_path, collection, topology_model_key=topology_model_key, force_local=force_local)
        else:
            raise ValueError(f"The path {pdf_or_folder_path} is neither a file nor a directory.")

        return collection
    
    @classmethod
    def extract_text_from_pdf(cls, pdf_path: str) -> List[str]:
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
                page_contents.append(cls.clean_pdf_text(text))
                converter.close()
                fake_file_handle.close()
                print(colored(f"{i+1}/{len(pages)}. Extracted page from '{pdf_path}'", "green"))
        return page_contents
    
    def clean_pdf_text(text: str):
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
        cache_dir = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "pdf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        last_modified = os.path.getmtime(file_path)
        full_cache_key = hashlib.md5(f"{file_path}_{last_modified}".encode()).hexdigest()
        return os.path.join(cache_dir, f"{cache_key}_{full_cache_key}.pickle")

    def load_from_cache(cache_file: str) -> Union[str, List[str], None]:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save_to_cache(cache_file: str, content: Union[str, List[str]]) -> None:
        with open(cache_file, 'wb') as f:
            pickle.dump(content, f)
    
    @classmethod
    def extract_pdf_content_page_wise(cls, file_path: str) -> List[str]:
        cache_file = cls.get_cache_file_path(file_path, "page_wise_text")
        cached_content = cls.load_from_cache(cache_file)
        
        if cached_content is not None:
            return cached_content
        
        page_contents = cls.extract_text_from_pdf(file_path)
        
        cls.save_to_cache(cache_file, page_contents)
        return page_contents

    @classmethod
    def rerank_results(
        cls,
        text_and_metas: List[Tuple[str, dict]],
        user_query: str,
        top_k: int = 3
    ) -> List[Tuple[str, dict]]:
        from sentence_transformers import CrossEncoder
        """
        Rerank the results using a reranker model.
        
        Args:
        text_and_metas (List[Tuple[str, dict]]): An iterable of tuples containing (document, metadata).
        user_query (str): The user query.
        top_k (int): Number of top results to return. Defaults to 3.
        
        Returns:
        List[Tuple[str, dict]]: The reranked and reduced results as a list of (document, metadata) tuples.
        """
        if len(text_and_metas) < top_k:
            top_k = len(text_and_metas)
        
        # Ensure we have results to work with
        if not text_and_metas:
            return text_and_metas  # Return original results if empty
        # Rerank the results using a reranker model
        reranker: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        rerank_pairs: List[List[str]] = [[user_query, doc] for doc, _ in text_and_metas]
        rerank_scores: List[float] = reranker.predict(rerank_pairs)
        # Combine original results with reranking scores
        reranked_results: List[Tuple[Tuple[str, dict], float]] = list(zip(text_and_metas, rerank_scores))
        # Sort results based on reranking scores (descending order)
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        # Get the top_k results
        top_results: List[Tuple[str, dict]] = [item[0] for item in reranked_results[:top_k]]
        
        return top_results

    @classmethod
    def create_rag_prompt(cls, text_and_metas: List[Tuple[str, Dict[str, str]]], user_query: str) -> str:
        """
        Create a RAG prompt from the reranked results.
        
        Args:
        text_and_metas (List[Tuple[str, Dict[str, str]]]): A list of (document, metadata) tuples.
        user_query (str): The user query.
        
        Returns:
        str: The generated RAG prompt.
        """
        # Create the retrieved context string
        prompt: str = "CONTEXT:\n"
        for i, (text, metadata) in enumerate(reversed(text_and_metas)):
            retrieved_context: str = "\n".join(text)
            retrieved_context = retrieved_context.replace("\n\n", "\n").strip()
            prompt += f"{i}. {retrieved_context}\n"

        prompt += f"\nINSTRUCTION:\nConsider the list from above and aim to provide a intelligent response to the user_query.\nUSER_QUERY: {user_query}"
        
        return prompt

    @classmethod
    def retrieve_augment(cls, user_query: str, collection: 'chromadb.Collection', top_k: int = 3) -> str:
        """
        Retrieve relevant information from a chroma collection based on a user query.

        Args:
            user_query (str): The user's query.
            collection (chromadb.Collection): The chroma collection to search.
            top_k (int, optional): The number of results to return. Defaults to 3.

        Returns:
            str: A RAG prompt containing the retrieved information.
        """
        # Create embedding for user query using Ollama
        user_query_embedding = OllamaClient.generate_embedding(user_query)
        
        # Query the collection for similar documents
        results = collection.query(
            query_embeddings=user_query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        # Zip documents and metadatas together for reranking
        docs_metas = list(zip(results['documents'][0], results['metadatas'][0]))
        
        # Rerank the results based on relevance to user query
        reranked_results = cls.rerank_results(docs_metas, user_query, top_k)
        
        # Create a RAG prompt from the reranked results
        rag_prompt = cls.create_rag_prompt(reranked_results, user_query)
        
        return rag_prompt

    @classmethod 
    def retrieve_augment_from_path(cls, user_query: str, path: str) -> str:
        """
        Retrieve augmented information from a PDF or folder path based on a user query.

        Args:
            user_query (str): The user's query.
            path (str): Path to a PDF file or folder containing PDFs.

        Returns:
            str: A RAG prompt containing the retrieved information.
        """
        # Import chromadb only when needed
        import chromadb
        
        # Create a client
        client = chromadb.Client()
        
        # Create a temporary collection
        collection = client.create_collection(name="temp_collection")
        
        # Process the PDF(s) and add to collection
        cls.pdf_or_folder_to_database(path, collection)
        
        # Use the collection to retrieve information
        return cls.retrieve_augment(user_query, collection)