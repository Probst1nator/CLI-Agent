# tools/searchtext.py
"""
This file implements the 'searchtext' tool. It allows the agent to perform
intelligent searches on large text resources managed by the TextResourceManager.

The tool supports:
- Regex pattern searching with context lines
- Line range retrieval 
- Resource listing and metadata inspection
- Efficient search across large files and process outputs

This tool is designed to work with resources created by enhanced readfile and
taskmanager tools that use the TextResourceManager for large content.
"""
import re
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False


class searchtext:
    """
    A tool to search and navigate large text resources.
    Provides intelligent search capabilities over files and process outputs
    that have been cached in the TextResourceManager.
    """

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'searchtext'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "searchtext",
            "description": (
                "The ONLY tool for inspecting large text content from files or process outputs. "
                "When `readfile` or `taskmanager` provides a `resource_id` because the content was too large, "
                "use this tool to search that content with a regex pattern or view specific line ranges."
            ),
            "example": (
                '<searchtext><resource_id>proc_a1b2c3d4</resource_id><pattern>error|failed</pattern><context_lines>3</context_lines></searchtext>\n'
                '<searchtext><resource_id>file_e5f6g7h8</resource_id><lines>100-200</lines></searchtext>'
            )
        }

    @staticmethod
    def _parse_xml_tags(content: str) -> Dict[str, str]:
        """Parse XML tags from content string."""
        tags = {}
        
        # Parse resource_id
        resource_match = re.search(r'<resource_id\s*>(.*?)</resource_id>', content, re.DOTALL | re.IGNORECASE)
        if resource_match:
            tags['resource_id'] = resource_match.group(1).strip()
        
        # Parse pattern for regex search
        pattern_match = re.search(r'<pattern\s*>(.*?)</pattern>', content, re.DOTALL | re.IGNORECASE)
        if pattern_match:
            tags['pattern'] = pattern_match.group(1).strip()
        
        # Parse context_lines
        context_match = re.search(r'<context_lines\s*>(.*?)</context_lines>', content, re.DOTALL | re.IGNORECASE)
        if context_match:
            try:
                tags['context_lines'] = int(context_match.group(1).strip())
            except ValueError:
                tags['context_lines'] = 0
        
        # Parse lines for range queries
        lines_match = re.search(r'<lines\s*>(.*?)</lines>', content, re.DOTALL | re.IGNORECASE)
        if lines_match:
            tags['lines'] = lines_match.group(1).strip()
        
        # Parse max_results
        max_match = re.search(r'<max_results\s*>(.*?)</max_results>', content, re.DOTALL | re.IGNORECASE)
        if max_match:
            try:
                tags['max_results'] = int(max_match.group(1).strip())
            except ValueError:
                tags['max_results'] = 50
        
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
    def _format_search_results(results: Dict[str, Any]) -> str:
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
    def _format_line_range_results(results: Dict[str, Any]) -> str:
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
    def _format_resource_list(resources: List[Dict[str, Any]]) -> str:
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
        
        output_lines.append("\nUse <searchtext><resource_id>RESOURCE_ID</resource_id><pattern>PATTERN</pattern></searchtext> to search a resource")
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
        
        # Check for list resources command
        if re.search(r'<list_resources\s*/>', content, re.IGNORECASE):
            resource_manager = TextResourceManager._get_manager()
            resources = resource_manager.list_resources()
            return searchtext._format_resource_list(resources)
        
        # Parse XML tags
        tags = searchtext._parse_xml_tags(content)
        
        resource_id = tags.get('resource_id')
        if not resource_id:
            return "❌ Error: No resource_id specified. Use <resource_id>RESOURCE_ID</resource_id> to specify which resource to search."
        
        # Check if resource exists using lazy access
        resource_manager = TextResourceManager._get_manager()
        metadata = resource_manager.get_metadata(resource_id)
        if not metadata:
            return f"❌ Error: Resource '{resource_id}' not found. Use <searchtext><list_resources/></searchtext> to see available resources."
        
        # Handle pattern search
        pattern = tags.get('pattern')
        if pattern:
            context_lines = tags.get('context_lines', 0)
            max_results = tags.get('max_results', 50)
            
            results = resource_manager.search_resource(
                resource_id=resource_id,
                pattern=pattern,
                context_lines=context_lines,
                max_results=max_results
            )
            return searchtext._format_search_results(results)
        
        # Handle line range query
        lines_spec = tags.get('lines')
        if lines_spec:
            try:
                start_line, end_line = searchtext._parse_line_range(lines_spec)
                results = resource_manager.get_line_range(
                    resource_id=resource_id,
                    start_line=start_line,
                    end_line=end_line
                )
                return searchtext._format_line_range_results(results)
            except ValueError:
                return f"❌ Error: Invalid line specification '{lines_spec}'. Use formats like '15', '10-20', '10-', or '-20'."
        
        # No specific operation specified - show resource info
        return f"📄 Resource Info: {resource_id}\n" \
               f"Source: {metadata['source']}\n" \
               f"Type: {metadata['source_type']}\n" \
               f"Lines: {metadata['total_lines']:,}\n" \
               f"Characters: {metadata['total_chars']:,}\n\n" \
               f"Use <searchtext><resource_id>{resource_id}</resource_id><pattern>SEARCH_PATTERN</pattern></searchtext> to search this resource.\n" \
               f"Use <searchtext><resource_id>{resource_id}</resource_id><lines>START-END</lines></searchtext> to view specific lines."