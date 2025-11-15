# tools/paper_search.py
"""
This file implements the 'paper_search' tool. It allows the agent to search
for academic papers on arXiv with advanced query syntax and filtering options.
"""
import asyncio
import os
import re
import sys
from datetime import datetime
from typing import Optional

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
    A tool to search for academic papers on arXiv with advanced query capabilities.
    Returns paper titles, authors, abstracts, and PDF links.
    """
    MAX_RESULTS = 10

    # Comprehensive help documentation
    HELP_TEXT = """PAPER_SEARCH - EXTENDED HELP

USAGE:
  <paper_search><query>machine learning transformers</query><max_results>5</max_results></paper_search>

PURPOSE:
Search academic papers using arXiv API. Returns title, abstract, authors, year, and PDF links.

PARAMETERS:
  <query> - Search terms (required, supports field syntax)
  <source> - arxiv|scholar|both (optional, default arxiv)
  <max_results> - 1-50 papers (optional, default 10)
  <sort_by> - relevance|submitted_date|updated_date (optional, default relevance)
  <sort_order> - ascending|descending (optional, default descending)
  <author> - Filter by author (optional)
  <category> - arXiv category filter like cs.AI, cs.LG (optional)
  <date_from> - Start date YYYY-MM-DD (optional)
  <date_to> - End date YYYY-MM-DD (optional)
  <arxiv_ids> - Comma-separated IDs for direct lookup (optional)

QUERY HANDLING:
  Simple queries (e.g., "artificial intelligence", "LLM", "transformers") are searched
  across multiple fields (title, abstract, authors, comments) by default.
  Advanced queries with field prefixes (ti:, au:, abs:) or boolean operators (AND, OR, ANDNOT)
  provide more precise control over search results.

EXAMPLES:
  Basic: <paper_search><query>neural networks</query></paper_search>
  Filtered: <paper_search><query>transformers</query><category>cs.LG</category><max_results>10</max_results></paper_search>
  Author: <paper_search><query>deep learning</query><author>Hinton</author></paper_search>
  Date range: <paper_search><query>attention</query><date_from>2017-01-01</date_from><date_to>2020-12-31</date_to></paper_search>

ADVANCED QUERY SYNTAX:
  Field searches:
    ti:title_words - Title only
    au:author_name - By author
    abs:abstract_words - Abstract
    cat:category - By category (cs.AI, cs.LG, etc.)

  Boolean operators (UPPERCASE):
    AND - Both required: au:Hinton AND ti:deep
    OR - Either: ti:transformer OR ti:attention
    ANDNOT - Exclude: cat:cs.AI ANDNOT ti:survey

COMMON CATEGORIES:
  cs.AI - Artificial Intelligence
  cs.LG - Machine Learning
  cs.CV - Computer Vision
  cs.CL - Computation and Language
  cs.NE - Neural Networks
  stat.ML - Statistics ML

RESULTS FORMAT:
  Title: Paper title
  Authors: Author list
  Published: YYYY-MM-DD
  Abstract: Summary (truncated)
  PDF: Download URL
  arXiv ID: Identifier

TIPS:
  • Use specific technical terms
  • Combine field searches: ti:transformer AND cat:cs.LG
  • Filter by recent dates for latest research
  • Use author field for specific researchers

LIMITATIONS:
  • Max 50 results per query
  • Rate limits apply
  • arXiv only (no other sources currently)
  • Some papers may lack PDF links

COMMON ERRORS:
  "No results" → Broaden query or check spelling
  "Invalid category" → Use format cs.XX or stat.XX
  "Rate limit" → Wait before retrying

RELATED: <help>websearch</help> <help>readfile</help>
"""

    @staticmethod
    def get_delim() -> str:
        return 'paper_search'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "paper_search",
            "description": (
                "Advanced academic paper search on arXiv with field-specific queries, boolean operators, "
                "and filtering. Supports author, category, date range filtering, and direct ID lookup. "
                "Use <help>paper_search</help> for comprehensive documentation. "
                "Basic: <query>search terms</query>. "
                "Advanced: <query>au:Hinton AND ti:deep</query><category>cs.AI</category><date_from>2023-01-01</date_from>. "
                "Direct lookup: <arxiv_ids>2103.14030,2010.11929</arxiv_ids>"
            ),
            "example": (
                "<paper_search><query>quantum computing</query></paper_search>\n"
                "<paper_search><query>au:Hinton AND ti:deep learning</query><max_results>5</max_results></paper_search>\n"
                "<paper_search><query>transformer</query><category>cs.LG</category><date_from>2023-01-01</date_from></paper_search>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return paper_search.HELP_TEXT

    @staticmethod
    def _build_arxiv_query(
        query: str,
        author: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Build an arXiv query with field prefixes and filters.

        Args:
            query: Base query string (may already contain field prefixes)
            author: Optional author name to add to query
            category: Optional category to add to query

        Returns:
            Enhanced query string with filters applied
        """
        query_parts = []

        # Add base query
        if query:
            # Pass query as-is - arXiv API searches across title, abstract, authors by default
            query_parts.append(query)

        # Add author filter if not already in query
        if author and 'au:' not in query.lower():
            query_parts.append(f'au:"{author}"')

        # Add category filter if not already in query
        if category and 'cat:' not in query.lower():
            query_parts.append(f'cat:{category}')

        # Join with AND operator
        final_query = ' AND '.join(query_parts) if len(query_parts) > 1 else query_parts[0] if query_parts else ''

        logger.debug(f"Built arXiv query: {final_query}")
        return final_query

    @staticmethod
    def _filter_by_date_range(
        papers: list[dict],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> list[dict]:
        """
        Filter papers by publication date range.

        Args:
            papers: List of paper dictionaries
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format

        Returns:
            Filtered list of papers
        """
        if not date_from and not date_to:
            return papers

        filtered = []
        for paper in papers:
            pub_date_str = paper.get('published', '')
            if not pub_date_str:
                continue

            try:
                # Parse publication date
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')

                # Check date range
                if date_from:
                    start_date = datetime.strptime(date_from, '%Y-%m-%d')
                    if pub_date < start_date:
                        continue

                if date_to:
                    end_date = datetime.strptime(date_to, '%Y-%m-%d')
                    if pub_date > end_date:
                        continue

                filtered.append(paper)
            except ValueError as e:
                logger.warning(f"Invalid date format for paper: {e}")
                continue

        logger.info(f"Date filtering: {len(papers)} → {len(filtered)} papers")
        return filtered

    @staticmethod
    async def _search_arxiv_by_ids(arxiv_ids: list[str]) -> tuple[list[dict], Optional[str]]:
        """
        Look up papers by arXiv IDs directly.

        Args:
            arxiv_ids: List of arXiv IDs (e.g., ['2103.14030', '2010.11929'])

        Returns:
            Tuple of (List of paper dictionaries, error message if any)
        """
        try:
            import arxiv

            logger.info(f"Looking up {len(arxiv_ids)} arXiv papers by ID")

            # Create search client
            client = arxiv.Client()

            # Build search with ID list
            search = arxiv.Search(id_list=arxiv_ids)

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
                    'comment': result.comment if hasattr(result, 'comment') else None,
                    'journal_ref': result.journal_ref if hasattr(result, 'journal_ref') else None,
                    'doi': result.doi if hasattr(result, 'doi') else None,
                    'source': 'arxiv'
                }
                papers.append(paper)

            logger.info(f"Retrieved {len(papers)} papers by ID")
            return papers, None

        except ImportError:
            error_msg = "The 'arxiv' library is not installed. Please install it with: pip install arxiv"
            logger.error(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"Error searching arXiv: {str(e)}"
            logger.error(f"Error looking up arXiv papers by ID: {e}", exc_info=True)
            return [], error_msg

    @staticmethod
    async def _search_arxiv(
        query: str,
        max_results: int,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> tuple[list[dict], Optional[str]]:
        """
        Search arXiv for papers using the official arXiv API.

        Args:
            query: Search query string (with field prefixes and operators)
            max_results: Maximum number of results to return
            sort_by: Sort criterion - 'relevance', 'submitted_date', or 'updated_date'
            sort_order: Sort order - 'ascending' or 'descending'

        Returns:
            Tuple of (List of paper dictionaries, error message if any)
        """
        try:
            import arxiv

            logger.info(f"Searching arXiv for: '{query}' (max={max_results}, sort={sort_by}/{sort_order})")

            # Map sort_by to arXiv API criteria
            sort_criterion_map = {
                'relevance': arxiv.SortCriterion.Relevance,
                'submitted_date': arxiv.SortCriterion.SubmittedDate,
                'date': arxiv.SortCriterion.SubmittedDate,  # Alias
                'updated_date': arxiv.SortCriterion.LastUpdatedDate
            }

            sort_criterion = sort_criterion_map.get(sort_by, arxiv.SortCriterion.Relevance)

            # Map sort_order to arXiv API
            sort_order_value = arxiv.SortOrder.Ascending if sort_order == 'ascending' else arxiv.SortOrder.Descending

            # Create search client
            client = arxiv.Client()

            # Build search query
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=sort_order_value
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
                    'comment': result.comment if hasattr(result, 'comment') else None,
                    'journal_ref': result.journal_ref if hasattr(result, 'journal_ref') else None,
                    'doi': result.doi if hasattr(result, 'doi') else None,
                    'source': 'arxiv'
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers on arXiv")
            return papers, None

        except ImportError:
            error_msg = "The 'arxiv' library is not installed. Please install it with: pip install arxiv"
            logger.error(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"Error searching arXiv: {str(e)}"
            logger.error(f"Error searching arXiv: {e}", exc_info=True)
            return [], error_msg

    @staticmethod
    async def _search_google_scholar(query: str, max_results: int, sort_by: str = "relevance") -> list[dict]:
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
    def _format_papers(papers: list[dict]) -> str:
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

            if paper.get('updated'):
                result_parts.append(f"  Updated: {paper['updated']}")

            # Add venue for Google Scholar results
            if paper.get('venue') and paper['venue'] != 'Unknown':
                result_parts.append(f"  Venue: {paper['venue']}")

            # Add citations for Google Scholar results
            if paper.get('citations'):
                result_parts.append(f"  Citations: {paper['citations']}")

            # Add categories for arXiv results
            if paper.get('primary_category'):
                result_parts.append(f"  Category: {paper['primary_category']}")

            if paper.get('categories') and len(paper['categories']) > 1:
                other_cats = [c for c in paper['categories'] if c != paper.get('primary_category')]
                if other_cats:
                    result_parts.append(f"  Other Categories: {', '.join(other_cats[:3])}")

            # Add DOI if available
            if paper.get('doi'):
                result_parts.append(f"  DOI: {paper['doi']}")

            # Add journal reference if available
            if paper.get('journal_ref'):
                result_parts.append(f"  Journal: {paper['journal_ref']}")

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

            # Add arXiv ID for easy reference
            if paper.get('arxiv_id'):
                result_parts.append(f"  arXiv ID: {paper['arxiv_id']}")

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
            sort_order_match = re.search(r'<sort_order>(.*?)</sort_order>', content, re.DOTALL)
            author_match = re.search(r'<author>(.*?)</author>', content, re.DOTALL)
            category_match = re.search(r'<category>(.*?)</category>', content, re.DOTALL)
            date_from_match = re.search(r'<date_from>(.*?)</date_from>', content, re.DOTALL)
            date_to_match = re.search(r'<date_to>(.*?)</date_to>', content, re.DOTALL)
            arxiv_ids_match = re.search(r'<arxiv_ids>(.*?)</arxiv_ids>', content, re.DOTALL)

            # Handle direct arXiv ID lookup
            if arxiv_ids_match:
                ids_str = arxiv_ids_match.group(1).strip()
                arxiv_ids = [id.strip() for id in ids_str.split(',') if id.strip()]

                if not arxiv_ids:
                    return "Error: No valid arXiv IDs provided."

                logger.info(f"Direct arXiv ID lookup: {arxiv_ids}")
                print(f"🔎 Looking up {len(arxiv_ids)} papers by arXiv ID...")

                papers, error = await paper_search._search_arxiv_by_ids(arxiv_ids)

                if error:
                    return f"Error: {error}"

                if not papers:
                    return f"No papers found for IDs: {', '.join(arxiv_ids)}"

                result = paper_search._format_papers(papers)
                print(f"✅ Found {len(papers)} papers")
                return result

            # Validate required parameters for query-based search
            if not query_match:
                return "Error: No search query provided. Use <query>your search</query> or <help>paper_search</help> for documentation."

            query = query_match.group(1).strip()
            if not query:
                return "Error: Query cannot be empty."

            # Parse optional parameters
            source = source_match.group(1).strip().lower() if source_match else 'arxiv'
            max_results = int(max_results_match.group(1)) if max_results_match else paper_search.MAX_RESULTS
            sort_by = sort_by_match.group(1).strip().lower() if sort_by_match else 'relevance'
            sort_order = sort_order_match.group(1).strip().lower() if sort_order_match else 'descending'
            author = author_match.group(1).strip() if author_match else None
            category = category_match.group(1).strip() if category_match else None
            date_from = date_from_match.group(1).strip() if date_from_match else None
            date_to = date_to_match.group(1).strip() if date_to_match else None

            # Validate parameters
            if source not in ['arxiv', 'scholar', 'both']:
                return f"Error: Invalid source '{source}'. Use 'arxiv', 'scholar', or 'both'."

            if max_results < 1 or max_results > 50:
                return "Error: max_results must be between 1 and 50."

            if sort_by not in ['relevance', 'date', 'submitted_date', 'updated_date']:
                return f"Error: Invalid sort_by '{sort_by}'. Use 'relevance', 'submitted_date', or 'updated_date'."

            if sort_order not in ['ascending', 'descending']:
                return f"Error: Invalid sort_order '{sort_order}'. Use 'ascending' or 'descending'."

            # Validate date formats
            if date_from:
                try:
                    datetime.strptime(date_from, '%Y-%m-%d')
                except ValueError:
                    return f"Error: Invalid date_from format '{date_from}'. Use YYYY-MM-DD."

            if date_to:
                try:
                    datetime.strptime(date_to, '%Y-%m-%d')
                except ValueError:
                    return f"Error: Invalid date_to format '{date_to}'. Use YYYY-MM-DD."

            # Build enhanced query for arXiv
            if source in ['arxiv', 'both']:
                enhanced_query = paper_search._build_arxiv_query(query, author, category)
            else:
                enhanced_query = query

            logger.info(f"Paper search: query='{enhanced_query}', source={source}, max={max_results}, sort={sort_by}/{sort_order}")

            # Build search description
            filters = []
            if author:
                filters.append(f"author: {author}")
            if category:
                filters.append(f"category: {category}")
            if date_from or date_to:
                date_range = f"{date_from or 'any'} to {date_to or 'any'}"
                filters.append(f"dates: {date_range}")

            filter_str = f" [{', '.join(filters)}]" if filters else ""
            print(f"🔎 Searching for papers: '{query}'{filter_str} (source: {source}, max: {max_results})")

            # Perform searches based on source parameter
            all_papers = []
            search_error = None

            if source in ['arxiv', 'both']:
                arxiv_papers, error = await paper_search._search_arxiv(enhanced_query, max_results, sort_by, sort_order)
                if error:
                    search_error = error
                else:
                    all_papers.extend(arxiv_papers)

            if source in ['scholar', 'both']:
                scholar_papers = await paper_search._search_google_scholar(query, max_results, sort_by)
                all_papers.extend(scholar_papers)

            # If there was an error and no papers from other sources, return the error
            if search_error and not all_papers:
                return f"Error: {search_error}"

            # Apply date filtering if specified
            if date_from or date_to:
                all_papers = paper_search._filter_by_date_range(all_papers, date_from, date_to)

            # Handle no results
            if not all_papers:
                return f"No papers found for query: '{query}'{filter_str}"

            # Sort combined results if searching both sources
            if source == 'both' and sort_by in ['date', 'submitted_date']:
                # Sort by publication date
                all_papers.sort(
                    key=lambda p: str(p.get('published', '0000-00-00')),
                    reverse=(sort_order == 'descending')
                )

            # Limit to max_results after combining and filtering
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
        print("=== TESTING BASIC SEARCH ===")
        basic_result = await paper_search.run('<query>quantum computing</query><max_results>2</max_results>')
        print(basic_result)

        print("\n=== TESTING ADVANCED SEARCH ===")
        advanced_result = await paper_search.run('<query>au:Hinton AND ti:deep</query><max_results>2</max_results>')
        print(advanced_result)

        print("\n=== TESTING DETAILED HELP ===")
        help_text = paper_search.get_detailed_tool_info()
        print(help_text[:500] + "...[truncated]")

    asyncio.run(test_search())
