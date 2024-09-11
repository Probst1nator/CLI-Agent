import json
import os
import os
import re
import pickle
import chromadb
import hashlib
from io import StringIO
from typing import Dict, Iterable, List, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime
from termcolor import colored
from sentence_transformers import CrossEncoder

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g


class RagTooling:
    @classmethod
    def pdf_or_folder_to_database(cls, pdf_or_folder_path: str, collection: chromadb.Collection, topology_model_key: str = "phi3.5", force_local: bool = True) -> chromadb.Collection:
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
        cache_dir = os.path.join(g.PROJ_VSCODE_DIR_PATH, "pdf_cache")
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
    def _process_single_pdf(cls, pdf_file_path: str, collection: chromadb.Collection, topology_model_key: str, force_local: bool = True) -> None:
        """
        Helper function to process a single PDF file.
        Args:
        pdf_file_path (str): The file path of the PDF to process.
        collection (chromadb.Collection): The collection to store the extractions in.
        """
        # Check if cache for the pdf file exists
        alle_informationen: List[str] = []
        file_cache_path = pdf_file_path + "_transformed.json"
        if os.path.exists(file_cache_path):
            alle_informationen = json.load(open(file_cache_path, "r"))
            for informations in alle_informationen:
                # Generate informationen embedding and add to vector database
                informations_hash = hashlib.md5(informations.encode()).hexdigest()
                # Add the content to the collection if it doesn't exist
                if not collection.get(informations_hash)['documents']:
                    informationen_embedding = OllamaClient.generate_embedding(informations)
                    collection.add(
                        ids=[informations_hash],
                        embeddings=informationen_embedding,
                        metadatas=[{"file_path": pdf_file_path, "file_name": file_name, "last_modified": last_modified, "source_text": coherent_extraction}],
                        documents=[informations]
                    )
            return
        
        
        file_name = os.path.basename(pdf_file_path).replace(" ", "_")
        last_modified = datetime.fromtimestamp(os.stat(pdf_file_path).st_mtime).isoformat()
        
        # list of strings, containing the text content of each page
        pages_extracted_content: List[str] = cls.extract_pdf_content_page_wise(pdf_file_path)
        print(colored(f"pages count:\t{len(pages_extracted_content)}", "yellow"))
        # let's always look at a window of 3 pages such that we can capture context accurately
        # we'll determine for each page if it better belongs to the previous or next page
        coherent_extractions: List[str] = []
        coherent_extraction_cache: str = ""
        for i in range(len(pages_extracted_content) - 2):
            print(colored(f"{i+1}/{len(pages_extracted_content)}. pages_extracted_content to coherent_extractions ", "green"))
            coherent_extraction_cache += pages_extracted_content[i]
            
            may_continue_on_next_page = True
            if len(pages_extracted_content) > i+1:
                # if "1 Modulbezeichnung" present on next page, then this is the last page of the coherent extraction
                if "1 Modulbezeichnung" in pages_extracted_content[i+1]:
                    may_continue_on_next_page = False
            # if "1 Modulbezeichnung" is not found anywhere later in the pages, switch to heuristical chunking
            remaining_pages: str = "".join(pages_extracted_content[i+2:])
            if may_continue_on_next_page: 
                if not "1 Modulbezeichnung" in remaining_pages:
                    # First heuristic
                    may_continue_on_next_page, yes_no_chat = FewShotProvider.few_shot_YesNo(f"If the following document is cut off abruptly at its end, respond with 'yes'. Otherwise, respond with 'no'.\n'''document\n{coherent_extraction_cache}\n'''", preferred_model_keys=["gemma2-9b-it"] + [topology_model_key], force_local = force_local, silent = True, force_free = True)
                    
                    # Second heuristic
                    if may_continue_on_next_page and i < len(pages_extracted_content) - 1:
                        yes_no_chat.add_message(Role.USER, f"This is the next page of the document, does it start a new topic/subject different to the previous page I showed you before? If a new topic/subject is started respond with 'yes', otherwise 'no'.\n'''document\n{pages_extracted_content[i+1]}\n'''")
                        is_next_page_new_topic, yes_no_chat = FewShotProvider.few_shot_YesNo(yes_no_chat, preferred_model_keys=["gemma2-9b-it"] + [topology_model_key], force_local = force_local, silent = True, force_free = True)
                        may_continue_on_next_page = not is_next_page_new_topic
                else:
                    # if "1 Modulbezeichnung" is found in the remaining pages [i+2:] and not in the next Page, then we can continue on the next page
                    may_continue_on_next_page = True
            if not may_continue_on_next_page:
                print(colored(f"Coherent extraction tokens:\t{len(coherent_extraction_cache)/3}", "yellow"))
                coherent_extractions.append(coherent_extraction_cache)
                coherent_extraction_cache = ""
        # # DEBUG
        # with open("filename.txt", 'w') as file:
        #     json.dump(coherent_extractions, file, indent=4)
            
        # Let's rephrase the coherent extractions into even more coherent chunks
        for i, coherent_extraction in enumerate(coherent_extractions):
            print(colored(f"{i+1}/{len(coherent_extractions)}. coherent_extraction to coherent_chunks", "cyan"))
            
            # Transform the extractable information to a german presentation
            chat = Chat()
            chat.add_message(Role.USER, f"The following text is an automated extraction from a PDF document. The PDF document was named '{file_name}'. Please reason shortly about it's contents and their context. Focus on explaining the relation between source, context and reliability of the content.\n\n'''\n{coherent_extraction}\n'''")
            high_level_extraction_analysis = LlmRouter.generate_completion(chat, preferred_model_keys=["llama3-70b-8192"], force_local = force_local, force_free = True, silent = True)
            chat.add_message(Role.ASSISTANT, high_level_extraction_analysis)
            chat.add_message(Role.USER, "Can you please summarize all details of the document in a coherent manner? The summary will be used to provide advice to students, this requires you to only provide facts that have plenty of context of topic and subject available. If such context is not present, always choose to skip unreliable or inaccurate information completely. Do not mention when you are ignoring content because of this.")
            factual_summarization = LlmRouter.generate_completion(chat, preferred_model_keys=["llama3-70b-8192"], force_local = force_local, silent = True, force_free = True)
            chat.add_message(Role.ASSISTANT, factual_summarization)
            praesentieren_prompt = "Please present the following information in a way that is easy to understand and in complete sentences. Begin directly with presenting.\n'''\n" + factual_summarization + "\n'''"
            chat.add_message(Role.USER, praesentieren_prompt)
            presented_information = LlmRouter.generate_completion(chat, preferred_model_keys=["llama-3.1-8b-instant"], force_local = force_local, silent = True, force_free = True)
            chat.add_message(Role.ASSISTANT, presented_information)
            # Because we're working with a very small model it often breaks, this we'll try alernate models until we give up and skip the information
            # We need to try models similar to the production model for the resulting ontology to fit optimally
            # Todo: Still waiting for phi3.5-moe to become available on ollama or as gguf on huggingface
            informations = LlmRouter.generate_completion(chat, preferred_model_keys=[topology_model_key], force_local = force_local, silent = False, force_free = True, force_preferred_model = False)
            # Safe guards for any issues that might ocurr
            informations_valid: bool = informations and not len(informations)>2048
            if informations_valid:
                is_understandable, _ = FewShotProvider.few_shot_YesNo(f"Is this understandable and readable? \n'''\n{informations}\n'''", preferred_model_keys=["gemma2-9b-it"] + [topology_model_key], force_local = force_local, silent = True, force_free = True)
                if is_understandable:
                    alle_informationen.append(informations)
                else:
                    print(colored("# # # A chunk was bad and could not be fixed, skipping... # # #", "red"))
                    print(colored(informations, "red"))
                    continue
            else:
                print(colored("# # # A chunk was bad and could not be fixed, skipping... # # #", "red"))
                print(colored(informations, "red"))
                continue
            
            # Generate informationen embedding and add to vector database
            informations_hash = hashlib.md5(informations.encode()).hexdigest()
            # Add the content to the collection if it doesn't exist
            if not collection.get(informations_hash)['documents']:
                informationen_embedding = OllamaClient.generate_embedding(informations)
                collection.add(
                    ids=[informations_hash],
                    embeddings=informationen_embedding,
                    metadatas=[{"file_path": pdf_file_path, "file_name": file_name, "last_modified": last_modified, "source_text": coherent_extraction}],
                    documents=[informations]
                )
        
        json.dump(alle_informationen, open(file_cache_path, "w"), indent=4)
        # # DEBUG
        # with open("filename.txt", 'w') as file:
        #     json.dump(coherent_chunks, file, indent=4)

    @classmethod
    def rerank_results(
        cls,
        text_and_metas: List[Tuple[str, dict]],
        user_query: str,
        top_k: int = 3
    ) -> List[Tuple[str, dict]]:
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
        if not text_and_metas:
            return "Respond by telling that the knowledge base is empty."
        # Group documents by source
        source_groups: Dict[str, List[str]] = defaultdict(list)
        for document, metadata in text_and_metas:
            source_groups[metadata['file_path']].append(document)
        # Create the retrieved context string
        prompt: str = ""
        for i, (source, documents) in enumerate(source_groups.items()):
            retrieved_context: str = "\n".join(documents)
            retrieved_context = retrieved_context.replace("\n\n", "\n").strip()
            prompt += f"Document:{i}\nSource: {source}\nText: {retrieved_context}\n\n"

        prompt += f"Instruction: Use the data from the above documents to provide a helpful and accurate response to the following question.\nQuestion: {user_query}"
        
        return prompt

    @classmethod
    def retrieve_augment(cls, user_query: str, collection: chromadb.Collection, top_k: int = 3) -> str:
        """
        Retrieve and Augment step for RAG
        
        Args:
        user_query (str): The user's query.
        collection (chromadb.Collection): The Chroma collection to query.
        
        Returns:
        str: The generated RAG prompt.
        """
        user_query_embedding: List[float] = OllamaClient.generate_embedding(user_query)
        results: Dict[str, Any] = collection.query(
            query_embeddings=user_query_embedding,
            n_results=top_k*3
        )
        text_and_metas = list(zip(results["documents"][0], results["metadatas"][0]))
        # rerank
        reranked_results: List[Tuple[str, Dict[str, str]]] = cls.rerank_results(text_and_metas, user_query, top_k)
        rag_prompt: str = cls.create_rag_prompt(reranked_results, user_query)
        return rag_prompt