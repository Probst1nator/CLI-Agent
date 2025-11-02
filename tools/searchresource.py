# tools/searchresource.py
"""
This file implements the 'searchresource' tool. It allows the agent to perform
intelligent searches on large text resources managed by the TextResourceManager.

The tool supports:
- Regex pattern searching with context lines
- Line range retrieval
- Resource listing and metadata inspection
- Efficient search across large files and process outputs

This tool is designed to work with resources created by enhanced readfile and
taskmanager tools that use the TextResourceManager for large content.
"""
import os
import re
import sys
from typing import Any

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False


class searchresource:
    """
    A tool to search and navigate large text resources.
    Provides intelligent search capabilities over files and process outputs
    that have been cached in the TextResourceManager.
    """

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'searchresource'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "searchresource",
            "description": (
                "Search and inspect large text resources cached by TextResourceManager. "
                "When `readfile`, `taskmanager`, or `readdirectory` provides a `resource_id` because content was too large, "
                "use this tool to search with regex patterns or retrieve specific line ranges. "
                "Supports GLOBAL SEARCH across all resources by omitting resource_id."
            ),
            "example": (
                '<searchresource><search>error|failed</search><context>3</context></searchresource>  # Global search\n'
                '<searchresource><resource_id>proc_abc123</resource_id><search>async.*error</search><context>2</context><limit>20</limit></searchresource>  # Search specific resource\n'
                '<searchresource><resource_id>file_xyz789</resource_id><lines>100-200</lines></searchresource>  # View line range\n'
                '<searchresource><list/></searchresource>  # List all resources'
            )
        }

    @staticmethod
    def _parse_xml_tags(content: str) -> dict[str, str]:
        """Parse XML tags from content string."""
        tags = {}

        # Parse resource_id
        resource_match = re.search(r'<resource_id\s*>(.*?)</resource_id>', content, re.DOTALL | re.IGNORECASE)
        if resource_match:
            tags['resource_id'] = resource_match.group(1).strip()

        # Parse search query (new: <search>, old: <pattern>)
        search_match = re.search(r'<search\s*>(.*?)</search>', content, re.DOTALL | re.IGNORECASE)
        if search_match:
            tags['search'] = search_match.group(1).strip()
        else:
            # Backward compatibility: <pattern>
            pattern_match = re.search(r'<pattern\s*>(.*?)</pattern>', content, re.DOTALL | re.IGNORECASE)
            if pattern_match:
                tags['search'] = pattern_match.group(1).strip()
                tags['_deprecated_pattern'] = True

        # Parse mode (regex, bm25, hybrid)
        mode_match = re.search(r'<mode\s*>(.*?)</mode>', content, re.DOTALL | re.IGNORECASE)
        if mode_match:
            tags['mode'] = mode_match.group(1).strip().lower()
        else:
            tags['mode'] = 'regex'  # Default to regex for backward compatibility

        # Parse context lines (new: <context>, old: <context_lines>)
        context_match = re.search(r'<context\s*>(.*?)</context>', content, re.DOTALL | re.IGNORECASE)
        if context_match:
            try:
                tags['context'] = int(context_match.group(1).strip())
            except ValueError:
                tags['context'] = 0
        else:
            # Backward compatibility: <context_lines>
            context_lines_match = re.search(r'<context_lines\s*>(.*?)</context_lines>', content, re.DOTALL | re.IGNORECASE)
            if context_lines_match:
                try:
                    tags['context'] = int(context_lines_match.group(1).strip())
                    tags['_deprecated_context_lines'] = True
                except ValueError:
                    tags['context'] = 0

        # Parse lines for range queries
        lines_match = re.search(r'<lines\s*>(.*?)</lines>', content, re.DOTALL | re.IGNORECASE)
        if lines_match:
            tags['lines'] = lines_match.group(1).strip()

        # Parse result limit (new: <limit>, old: <max_results>)
        limit_match = re.search(r'<limit\s*>(.*?)</limit>', content, re.DOTALL | re.IGNORECASE)
        if limit_match:
            try:
                tags['limit'] = int(limit_match.group(1).strip())
            except ValueError:
                tags['limit'] = 50
        else:
            # Backward compatibility: <max_results>
            max_match = re.search(r'<max_results\s*>(.*?)</max_results>', content, re.DOTALL | re.IGNORECASE)
            if max_match:
                try:
                    tags['limit'] = int(max_match.group(1).strip())
                    tags['_deprecated_max_results'] = True
                except ValueError:
                    tags['limit'] = 50

        return tags

    @staticmethod
    def _parse_line_range(line_spec: str) -> tuple:
        """
        Parse line specification into (start_line, end_line).

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
    def _format_search_results(results: dict[str, Any]) -> str:
        """Format search results for display."""
        if "error" in results:
            return f"❌ Search Error: {results['error']}"

        matches = results.get('matches', [])
        if not matches:
            return f"🔍 No matches found for pattern '{results.get('pattern', '')}' in resource {results.get('resource_id', 'unknown')}"

        # Build formatted output
        output_lines = []
        output_lines.append(f"🔍 Search Results for '{results['pattern']}' in {results.get('source_type', 'resource')} '{results.get('source', results.get('resource_id'))}'\n")
        output_lines.append(f"Found {results['total_matches']} matches:\n")

        for i, match in enumerate(matches[:20], 1):  # Show first 20 matches
            output_lines.append(f"Match {i} (Line {match['line_number']}):")

            # Show context if available
            context = match.get('context', [])
            if context:
                for ctx in context:
                    line_num = ctx['line_number']
                    content = ctx['content']
                    is_match = ctx['is_match']

                    # Truncate very long lines
                    if len(content) > 200:
                        content = content[:200] + "..."

                    if is_match:
                        output_lines.append(f"  >>> {line_num:4d}: {content}")
                    else:
                        output_lines.append(f"      {line_num:4d}: {content}")
            else:
                # Fallback to match preview
                preview = match.get('match_preview', match.get('matched_line', ''))
                output_lines.append(f"  >>> {match['line_number']:4d}: {preview}")

            output_lines.append("")  # Empty line between matches

        if results['total_matches'] > 20:
            output_lines.append(f"... and {results['total_matches'] - 20} more matches. Use more specific patterns to narrow results.")

        return "\n".join(output_lines)

    @staticmethod
    def _format_global_search_results(search_query: str, all_results: list[dict[str, Any]]) -> str:
        """Format global search results across multiple resources."""
        total_matches = sum(r['total_matches'] for r in all_results if 'total_matches' in r)
        resources_with_matches = [r for r in all_results if r.get('total_matches', 0) > 0]

        if not resources_with_matches:
            return f"🔍 Global Search: No matches found for search '{search_query}' across {len(all_results)} resources"

        # Build formatted output
        output_lines = []
        output_lines.append(f"🌐 Global Search Results for '{search_query}'")
        output_lines.append(f"Found {total_matches} total matches across {len(resources_with_matches)} resources:\n")

        for resource_result in resources_with_matches[:10]:  # Show first 10 resources with matches
            source = resource_result.get('source', resource_result.get('resource_id', 'unknown'))
            matches = resource_result.get('matches', [])
            total = resource_result['total_matches']

            output_lines.append(f"📄 {source} ({total} matches):")

            # Show first 3 matches per resource
            for i, match in enumerate(matches[:3], 1):
                line_num = match['line_number']
                preview = match.get('match_preview', match.get('matched_line', ''))

                # Truncate very long lines
                if len(preview) > 150:
                    preview = preview[:150] + "..."

                output_lines.append(f"  >>> Line {line_num}: {preview}")

            if total > 3:
                output_lines.append(f"  ... and {total - 3} more matches in this resource")

            output_lines.append("")  # Empty line between resources

        if len(resources_with_matches) > 10:
            output_lines.append(f"... and {len(resources_with_matches) - 10} more resources with matches")

        output_lines.append(f"\n💡 Use <searchresource><resource_id>RESOURCE_ID</resource_id><search>{search_query}</search></searchresource> to search a specific resource")

        return "\n".join(output_lines)

    @staticmethod
    def _format_line_range_results(results: dict[str, Any]) -> str:
        """Format line range results for display."""
        if "error" in results:
            return f"❌ Line Range Error: {results['error']}"

        lines = results.get('lines', [])
        if not lines:
            return "📄 No lines found in specified range"

        # Build formatted output
        output_lines = []
        source = results.get('source', results.get('resource_id', 'unknown'))
        output_lines.append(f"📄 Lines {results['start_line']}-{results['end_line']} from {source}:\n")

        for line_info in lines:
            line_num = line_info['line_number']
            content = line_info['content']
            output_lines.append(f"{line_num:4d}→{content}")

        return "\n".join(output_lines)

    @staticmethod
    def _format_resource_list(resources: list[dict[str, Any]]) -> str:
        """Format resource list for display."""
        if not resources:
            return "📋 No text resources are currently available"

        output_lines = []
        output_lines.append("📋 Available Text Resources:\n")
        output_lines.append("Resource ID        | Source Type | Lines  | Chars   | Source")
        output_lines.append("-------------------|-------------|--------|---------|----------------------------------------")

        for resource in resources:
            resource_id = resource['resource_id'][:18]  # Truncate long IDs
            source_type = resource['source_type'][:10]
            lines = f"{resource['total_lines']:,}"
            chars = f"{resource['total_chars']:,}"
            source = resource['source']

            # Truncate long sources
            if len(source) > 40:
                source = source[:37] + "..."

            output_lines.append(f"{resource_id:<18} | {source_type:<11} | {lines:>6} | {chars:>7} | {source}")

        output_lines.append("\nUse <searchresource><resource_id>RESOURCE_ID</resource_id><search>PATTERN</search></searchresource> to search a resource")
        return "\n".join(output_lines)

    @staticmethod
    def run(content: str) -> str:
        """
        Execute search operations on text resources.

        Args:
            content: XML string containing search parameters

        Returns:
            Formatted search results or error message
        """
        if not RESOURCE_MANAGER_AVAILABLE:
            return "❌ Error: TextResourceManager not available. This tool requires the text resource management system."

        # Check for list resources command (both new <list/> and old <list_resources/>)
        if re.search(r'<list\s*/>', content, re.IGNORECASE) or re.search(r'<list_resources\s*/>', content, re.IGNORECASE):
            try:
                resource_manager = TextResourceManager._get_manager()
                resources = resource_manager.list_resources()
                return searchresource._format_resource_list(resources)
            except NameError:
                return "❌ Error: TextResourceManager not available. This tool requires the text resource management system."

        # Parse XML tags
        tags = searchresource._parse_xml_tags(content)

        resource_id = tags.get('resource_id')
        search_query = tags.get('search')  # New parameter name

        try:
            resource_manager = TextResourceManager._get_manager()
        except NameError:
            return "❌ Error: TextResourceManager not available. This tool requires the text resource management system."

        # Handle GLOBAL SEARCH (no resource_id specified, but search query given)
        if not resource_id and search_query:
            context = int(tags.get('context', 0))
            limit = int(tags.get('limit', 50))

            # Get all available resources
            all_resources = resource_manager.list_resources()
            if not all_resources:
                return "🔍 Global Search: No resources available to search"

            # Search each resource
            all_results = []
            for resource in all_resources:
                rid = resource['resource_id']
                try:
                    results = resource_manager.search_resource(
                        resource_id=rid,
                        pattern=search_query,
                        context_lines=context,
                        max_results=limit
                    )
                    all_results.append(results)
                except Exception:
                    # Skip resources that fail to search
                    continue

            return searchresource._format_global_search_results(search_query, all_results)

        # Require resource_id for non-search operations
        if not resource_id:
            return "❌ Error: No resource_id specified. Use <resource_id>RESOURCE_ID</resource_id> to specify which resource to search, or omit it for global search with a search query."

        # Check if resource exists
        metadata = resource_manager.get_metadata(resource_id)
        if not metadata:
            return f"❌ Error: Resource '{resource_id}' not found. Use <searchresource><list/></searchresource> to see available resources."

        # Handle search on specific resource
        if search_query:
            context = int(tags.get('context', 0))
            limit = int(tags.get('limit', 50))

            results = resource_manager.search_resource(
                resource_id=resource_id,
                pattern=search_query,
                context_lines=context,
                max_results=limit
            )
            return searchresource._format_search_results(results)

        # Handle line range query
        lines_spec = tags.get('lines')
        if lines_spec:
            try:
                start_line, end_line = searchresource._parse_line_range(lines_spec)
                results = resource_manager.get_line_range(
                    resource_id=resource_id,
                    start_line=start_line,
                    end_line=end_line
                )
                return searchresource._format_line_range_results(results)
            except ValueError:
                return f"❌ Error: Invalid line specification '{lines_spec}'. Use formats like '15', '10-20', '10-', or '-20'."

        # No specific operation specified - show resource info
        return f"📄 Resource Info: {resource_id}\n" \
               f"Source: {metadata['source']}\n" \
               f"Type: {metadata['source_type']}\n" \
               f"Lines: {metadata['total_lines']:,}\n" \
               f"Characters: {metadata['total_chars']:,}\n\n" \
               f"Use <searchresource><resource_id>{resource_id}</resource_id><search>SEARCH_PATTERN</search></searchresource> to search this resource.\n" \
               f"Use <searchresource><resource_id>{resource_id}</resource_id><lines>START-END</lines></searchresource> to view specific lines."