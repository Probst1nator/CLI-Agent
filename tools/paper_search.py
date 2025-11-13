# tools/paper_search.py
"""
This file implements the 'paper_search' tool. It allows the agent to search
for academic papers on arXiv and optionally Google Scholar.
"""
import asyncio
import os
import re
import sys
from typing import List, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure proper imports when running as module or script
if __name__ == '__main__':
    # Add project root to path for direct execution
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)


class paper_search:
    """
    A tool to search for academic papers on arXiv and Google Scholar.
    Returns paper titles, authors, abstracts, and PDF links.
    """
    MAX_RESULTS = 10

    @staticmethod
    def get_delim() -> str:
        return 'paper_search'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "paper_search",
            "description": (
                f"Searches for academic papers on arXiv and Google Scholar. "
                f"Requires <query> tag with search terms. Optional tags: "
                f"<source>arxiv|scholar|both</source> (default: both), "
                f"<max_results>N</max_results> (default: {paper_search.MAX_RESULTS}), "
                f"<sort_by>relevance|date</sort_by> (default: relevance). "
                f"Returns paper titles, authors, abstracts, publication dates, and PDF/paper links."
            ),
            "example": (
                "<paper_search><query>quantum computing</query></paper_search>\n"
                "<paper_search><query>machine learning transformers</query><source>arxiv</source><max_results>5</max_results></paper_search>\n"
                "<paper_search><query>neural networks</query><source>scholar</source><sort_by>date</sort_by></paper_search>"
            )
        }

    @staticmethod
    async def _search_arxiv(query: str, max_results: int, sort_by: str = "relevance") -> List[Dict]:
        """
        Search arXiv for papers using the official arXiv API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort order - 'relevance' or 'date'

        Returns:
            List of paper dictionaries with metadata
        """
        try:
            import arxiv

            logger.info(f"Searching arXiv for: '{query}' (max_results={max_results}, sort_by={sort_by})")

            # Determine sort order
            sort_order = arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate

            # Create search client
            client = arxiv.Client()

            # Build search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_order
            )

            # Execute search in thread pool to avoid blocking
            results = await asyncio.to_thread(list, client.results(search))

            papers = []
            for result in results:
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.strip(),
                    'published': result.published.strftime('%Y-%m-%d'),
                    'updated': result.updated.strftime('%Y-%m-%d') if result.updated else None,
                    'pdf_url': result.pdf_url,
                    'arxiv_url': result.entry_id,
                    'arxiv_id': result.get_short_id(),
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'source': 'arxiv'
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers on arXiv")
            return papers

        except ImportError:
            logger.error("arxiv library not installed. Install with: pip install arxiv")
            return []
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}", exc_info=True)
            return []

    @staticmethod
    async def _search_google_scholar(query: str, max_results: int, sort_by: str = "relevance") -> List[Dict]:
        """
        Search Google Scholar for papers.

        Note: Google Scholar actively blocks automated scraping. This uses the scholarly library
        which implements various anti-blocking measures, but may still be unreliable.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort order - 'relevance' or 'date'

        Returns:
            List of paper dictionaries with metadata
        """
        try:
            from scholarly import scholarly

            logger.info(f"Searching Google Scholar for: '{query}' (max_results={max_results})")
            logger.warning("Google Scholar may block automated requests. Results may be limited.")

            # Execute search in thread pool to avoid blocking
            search_query = await asyncio.to_thread(scholarly.search_pubs, query)

            papers = []
            count = 0

            # Iterate through results
            for result in search_query:
                if count >= max_results:
                    break

                try:
                    # Extract paper information
                    paper = {
                        'title': result.get('bib', {}).get('title', 'N/A'),
                        'authors': result.get('bib', {}).get('author', []),
                        'abstract': result.get('bib', {}).get('abstract', 'No abstract available'),
                        'published': result.get('bib', {}).get('pub_year', 'Unknown'),
                        'venue': result.get('bib', {}).get('venue', 'Unknown'),
                        'citations': result.get('num_citations', 0),
                        'url': result.get('pub_url', result.get('eprint_url', 'N/A')),
                        'pdf_url': result.get('eprint_url', 'N/A'),
                        'source': 'google_scholar'
                    }
                    papers.append(paper)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error parsing Google Scholar result: {e}")
                    continue

            logger.info(f"Found {len(papers)} papers on Google Scholar")
            return papers

        except ImportError:
            logger.error("scholarly library not installed. Install with: pip install scholarly")
            return []
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}", exc_info=True)
            return []

    @staticmethod
    def _format_papers(papers: List[Dict]) -> str:
        """
        Format paper results into a readable string.

        Args:
            papers: List of paper dictionaries

        Returns:
            Formatted string with paper information
        """
        if not papers:
            return "No papers found."

        formatted_results = []

        for i, paper in enumerate(papers, 1):
            source = paper.get('source', 'unknown')
            title = paper.get('title', 'No Title')
            authors = paper.get('authors', [])

            # Format authors
            if isinstance(authors, list):
                if len(authors) > 3:
                    authors_str = f"{', '.join(authors[:3])}, et al."
                else:
                    authors_str = ', '.join(authors)
            else:
                authors_str = str(authors)

            # Build result string
            result_parts = [
                f"Paper {i} [{source.upper()}]:",
                f"  Title: {title}",
                f"  Authors: {authors_str}",
            ]

            # Add publication info
            if paper.get('published'):
                result_parts.append(f"  Published: {paper['published']}")

            # Add venue for Google Scholar results
            if paper.get('venue') and paper['venue'] != 'Unknown':
                result_parts.append(f"  Venue: {paper['venue']}")

            # Add citations for Google Scholar results
            if paper.get('citations'):
                result_parts.append(f"  Citations: {paper['citations']}")

            # Add categories for arXiv results
            if paper.get('primary_category'):
                result_parts.append(f"  Category: {paper['primary_category']}")

            # Add abstract (truncated if too long)
            abstract = paper.get('abstract', 'No abstract available')
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            result_parts.append(f"  Abstract: {abstract}")

            # Add URLs
            if paper.get('pdf_url') and paper['pdf_url'] != 'N/A':
                result_parts.append(f"  PDF: {paper['pdf_url']}")

            if paper.get('arxiv_url'):
                result_parts.append(f"  arXiv: {paper['arxiv_url']}")
            elif paper.get('url') and paper['url'] != 'N/A':
                result_parts.append(f"  URL: {paper['url']}")

            formatted_results.append('\n'.join(result_parts))

        return '\n\n' + '\n\n'.join(formatted_results)

    @staticmethod
    async def run(content: str) -> str:
        """
        Executes a paper search based on the provided query and parameters.

        Args:
            content: XML-formatted string with search parameters

        Returns:
            Formatted string with paper search results
        """
        try:
            # Parse XML parameters
            query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
            source_match = re.search(r'<source>(.*?)</source>', content, re.DOTALL)
            max_results_match = re.search(r'<max_results>(\d+)</max_results>', content, re.DOTALL)
            sort_by_match = re.search(r'<sort_by>(.*?)</sort_by>', content, re.DOTALL)

            # Validate required parameters
            if not query_match:
                logger.error("No query provided in paper_search request")
                return "Error: No search query provided in the <query> tag."

            query = query_match.group(1).strip()
            if not query:
                return "Error: Query cannot be empty."

            # Parse optional parameters
            source = source_match.group(1).strip().lower() if source_match else 'both'
            max_results = int(max_results_match.group(1)) if max_results_match else paper_search.MAX_RESULTS
            sort_by = sort_by_match.group(1).strip().lower() if sort_by_match else 'relevance'

            # Validate parameters
            if source not in ['arxiv', 'scholar', 'both']:
                return f"Error: Invalid source '{source}'. Use 'arxiv', 'scholar', or 'both'."

            if max_results < 1 or max_results > 50:
                return "Error: max_results must be between 1 and 50."

            if sort_by not in ['relevance', 'date']:
                return f"Error: Invalid sort_by '{sort_by}'. Use 'relevance' or 'date'."

            logger.info(f"Paper search: query='{query}', source={source}, max_results={max_results}, sort_by={sort_by}")
            print(f"🔎 Searching for papers: '{query}' (source: {source}, max: {max_results})")

            # Perform searches based on source parameter
            all_papers = []

            if source in ['arxiv', 'both']:
                arxiv_papers = await paper_search._search_arxiv(query, max_results, sort_by)
                all_papers.extend(arxiv_papers)

            if source in ['scholar', 'both']:
                scholar_papers = await paper_search._search_google_scholar(query, max_results, sort_by)
                all_papers.extend(scholar_papers)

            # Handle no results
            if not all_papers:
                return f"No papers found for query: '{query}'"

            # Sort combined results if searching both sources
            if source == 'both' and sort_by == 'date':
                # Sort by publication date (newest first)
                all_papers.sort(
                    key=lambda p: str(p.get('published', '0000-00-00')),
                    reverse=True
                )

            # Limit to max_results after combining
            all_papers = all_papers[:max_results]

            # Format and return results
            result = paper_search._format_papers(all_papers)

            logger.info(f"Returning {len(all_papers)} papers to user")
            print(f"✅ Found {len(all_papers)} papers")

            return result

        except Exception as e:
            logger.error(f"Unexpected error in paper_search tool: {e}", exc_info=True)
            return f"An unexpected error occurred in the paper_search tool: {str(e)}"


# Allow direct execution for testing
if __name__ == '__main__':
    async def test_search():
        """Test the paper search tool"""
        test_query = '<query>quantum computing</query><source>arxiv</source><max_results>3</max_results>'
        result = await paper_search.run(test_query)
        print(result)

    asyncio.run(test_search())
