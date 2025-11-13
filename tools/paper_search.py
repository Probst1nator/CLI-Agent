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
    A tool to search for academic papers on arXiv with advanced query capabilities.
    Returns paper titles, authors, abstracts, and PDF links.
    """
    MAX_RESULTS = 10

    # Comprehensive help documentation
    HELP_TEXT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PAPER_SEARCH TOOL - EXTENDED HELP                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <paper_search>
    <query>your search terms</query>
  </paper_search>

PARAMETERS:
  <query>       - Search terms (supports advanced syntax, see below) [REQUIRED]
  <source>      - arxiv|scholar|both (default: arxiv) [OPTIONAL]
  <max_results> - 1-50 papers (default: 10) [OPTIONAL]
  <sort_by>     - relevance|submitted_date|updated_date (default: relevance) [OPTIONAL]
  <sort_order>  - ascending|descending (default: descending) [OPTIONAL]
  <author>      - Filter by author name (e.g., "Hinton") [OPTIONAL]
  <category>    - Filter by arXiv category (e.g., "cs.AI", "cs.LG") [OPTIONAL]
  <date_from>   - Start date YYYY-MM-DD (e.g., "2023-01-01") [OPTIONAL]
  <date_to>     - End date YYYY-MM-DD (e.g., "2024-12-31") [OPTIONAL]
  <arxiv_ids>   - Comma-separated arXiv IDs for direct lookup [OPTIONAL]

ADVANCED QUERY SYNTAX:
  Field-specific searches:
    ti:title_words          - Search in title only
    au:author_name          - Search by author
    abs:abstract_words      - Search in abstract
    cat:category            - Search by category (cs.AI, cs.LG, quant-ph, etc.)
    co:comment              - Search in comments
    jr:journal              - Search by journal reference
    all:keywords            - Search all fields (default)

  Boolean operators (use uppercase):
    AND     - Both terms required: au:Hinton AND ti:deep
    OR      - Either term: ti:transformer OR ti:attention
    ANDNOT  - Exclude: cat:cs.AI ANDNOT ti:survey

  Examples of complex queries:
    "au:LeCun AND ti:convolution"
    "cat:cs.LG AND abs:reinforcement"
    "ti:transformer ANDNOT ti:survey"
    "au:Hinton OR au:Bengio"

ARXIV CATEGORIES (common):
  Computer Science:
    cs.AI - Artificial Intelligence
    cs.LG - Machine Learning
    cs.CL - Computation and Language (NLP)
    cs.CV - Computer Vision
    cs.NE - Neural and Evolutionary Computing
    cs.RO - Robotics

  Physics:
    quant-ph - Quantum Physics
    cond-mat - Condensed Matter
    astro-ph - Astrophysics

  Mathematics:
    math.CO - Combinatorics
    math.ST - Statistics Theory

  Statistics:
    stat.ML - Machine Learning (Statistics)

USAGE EXAMPLES:

1. Basic keyword search:
   <paper_search>
     <query>quantum computing</query>
   </paper_search>

2. Search by specific author:
   <paper_search>
     <query>au:Hinton</query>
     <max_results>10</max_results>
   </paper_search>

3. Category-specific search:
   <paper_search>
     <query>deep learning</query>
     <category>cs.LG</category>
     <sort_by>submitted_date</sort_by>
     <sort_order>descending</sort_order>
   </paper_search>

4. Date range filtering:
   <paper_search>
     <query>transformer architecture</query>
     <date_from>2023-01-01</date_from>
     <date_to>2024-01-01</date_to>
   </paper_search>

5. Combined filters:
   <paper_search>
     <query>ti:reinforcement AND abs:policy</query>
     <author>Sutton</author>
     <category>cs.LG</category>
     <date_from>2020-01-01</date_from>
     <max_results>5</max_results>
   </paper_search>

6. Direct arXiv ID lookup:
   <paper_search>
     <arxiv_ids>2103.14030,2010.11929</arxiv_ids>
   </paper_search>

7. Complex boolean query:
   <paper_search>
     <query>cat:cs.AI AND (ti:transformer OR ti:attention) ANDNOT ti:survey</query>
     <max_results>15</max_results>
   </paper_search>

8. Author collaboration search:
   <paper_search>
     <query>au:Bengio AND au:LeCun</query>
   </paper_search>

TIPS:
  • Use field prefixes (ti:, au:, etc.) for more precise searches
  • Combine filters for better targeting
  • Sort by submitted_date for latest papers
  • Use ANDNOT to exclude review papers or surveys
  • Check paper categories to find related work
  • Direct ID lookup is fastest when you know the arXiv IDs

LIMITATIONS:
  • Google Scholar support is optional (may be unreliable)
  • Max 50 results per query
  • Date filtering applies post-search (may affect result count)
  • Complex queries may take longer to process

For more information on arXiv categories:
  https://arxiv.org/category_taxonomy

╚══════════════════════════════════════════════════════════════════════════════╝
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
                f"Advanced academic paper search on arXiv with field-specific queries, boolean operators, "
                f"and filtering. Supports author, category, date range filtering, and direct ID lookup. "
                f"Use <help>paper_search</help> for comprehensive documentation. "
                f"Basic: <query>search terms</query>. "
                f"Advanced: <query>au:Hinton AND ti:deep</query><category>cs.AI</category><date_from>2023-01-01</date_from>. "
                f"Direct lookup: <arxiv_ids>2103.14030,2010.11929</arxiv_ids>"
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
        papers: List[Dict],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict]:
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
    async def _search_arxiv_by_ids(arxiv_ids: List[str]) -> List[Dict]:
        """
        Look up papers by arXiv IDs directly.

        Args:
            arxiv_ids: List of arXiv IDs (e.g., ['2103.14030', '2010.11929'])

        Returns:
            List of paper dictionaries with metadata
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
            return papers

        except ImportError:
            logger.error("arxiv library not installed. Install with: pip install arxiv")
            return []
        except Exception as e:
            logger.error(f"Error looking up arXiv papers by ID: {e}", exc_info=True)
            return []

    @staticmethod
    async def _search_arxiv(
        query: str,
        max_results: int,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[Dict]:
        """
        Search arXiv for papers using the official arXiv API.

        Args:
            query: Search query string (with field prefixes and operators)
            max_results: Maximum number of results to return
            sort_by: Sort criterion - 'relevance', 'submitted_date', or 'updated_date'
            sort_order: Sort order - 'ascending' or 'descending'

        Returns:
            List of paper dictionaries with metadata
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

                papers = await paper_search._search_arxiv_by_ids(arxiv_ids)

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

            if source in ['arxiv', 'both']:
                arxiv_papers = await paper_search._search_arxiv(enhanced_query, max_results, sort_by, sort_order)
                all_papers.extend(arxiv_papers)

            if source in ['scholar', 'both']:
                scholar_papers = await paper_search._search_google_scholar(query, max_results, sort_by)
                all_papers.extend(scholar_papers)

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
