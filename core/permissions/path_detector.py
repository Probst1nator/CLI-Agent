# path_detector.py
"""
Path detection utility for XML blocks in CLI-Agent tools.
Detects file paths in XML tags containing 'path' substring and determines tool actions.
"""
import re
import os
from typing import Dict, List, Tuple, Optional

class PathDetector:
    # Define permission groups for related tools
    PERMISSION_GROUPS = {
        'file_editing': ['editfile', 'writefile', 'appendfile'],
        'file_reading': ['readfile', 'viewfiles'],
        'code_execution': ['bash', 'python', 'shell'],
        'file_management': ['bash', 'python']  # Only when they contain file operations
    }
    
    # Reverse mapping: tool -> groups it belongs to
    TOOL_TO_GROUPS = {}
    for group, tools in PERMISSION_GROUPS.items():
        for tool in tools:
            if tool not in TOOL_TO_GROUPS:
                TOOL_TO_GROUPS[tool] = []
            TOOL_TO_GROUPS[tool].append(group)
    
    @staticmethod
    def get_tool_groups(tool_name: str) -> List[str]:
        """Get all permission groups that a tool belongs to."""
        return PathDetector.TOOL_TO_GROUPS.get(tool_name, [])
    
    @staticmethod
    def get_group_tools(group_name: str) -> List[str]:
        """Get all tools that belong to a permission group."""
        return PathDetector.PERMISSION_GROUPS.get(group_name, [])
    
    @staticmethod
    def extract_paths_from_xml(xml_content: str, tool_name: str) -> List[Tuple[str, str]]:
        """
        Extract file paths from XML content by finding tags containing 'path'.
        
        Args:
            xml_content: XML content to parse
            tool_name: Name of the tool being executed
            
        Returns:
            List of (tag_name, file_path) tuples found in the XML
        """
        paths = []
        
        # Find all XML tags that contain 'path' in their name
        path_tag_pattern = r'<([^>]*path[^>]*)>(.*?)</\1>'
        matches = re.finditer(path_tag_pattern, xml_content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            tag_name = match.group(1).strip()
            content = match.group(2).strip()
            
            # Only consider non-empty content as potential file paths
            if content:
                # Resolve to absolute path
                abs_path = os.path.abspath(content)
                paths.append((tag_name, abs_path))
        
        return paths
    
    @staticmethod
    def get_tool_action_description(tool_name: str, xml_content: str, paths: List[Tuple[str, str]]) -> str:
        """
        Generate a human-readable description of what the tool will do.
        
        Args:
            tool_name: Name of the tool
            xml_content: Full XML content
            paths: List of (tag_name, file_path) tuples
            
        Returns:
            Human-readable action description
        """
        if not paths:
            return f"Execute {tool_name} (no file paths detected)"
        
        # Tool-specific action descriptions
        if tool_name == "editfile":
            find_match = re.search(r'<find>(.*?)</find>', xml_content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', xml_content, re.DOTALL)
            
            find_preview = find_match.group(1)[:50] + "..." if find_match and len(find_match.group(1)) > 50 else find_match.group(1) if find_match else "?"
            replace_preview = replace_match.group(1)[:50] + "..." if replace_match and len(replace_match.group(1)) > 50 else replace_match.group(1) if replace_match else "?"
            
            return f"Replace '{find_preview}' with '{replace_preview}'"
            
        elif tool_name == "writefile":
            return "Write/overwrite file content"
            
        elif tool_name == "readfile":
            return "Read file content"
            
        elif tool_name == "bash":
            # Check if bash commands involve file operations
            bash_content = xml_content.lower()
            if any(cmd in bash_content for cmd in ['rm ', 'delete', 'mv ', 'cp ', 'chmod', 'chown']):
                return "Execute potentially file-modifying bash commands"
            else:
                return "Execute bash commands"
                
        elif tool_name == "python":
            # Check for file operations in Python code
            python_content = xml_content.lower()
            if any(op in python_content for op in ['open(', 'write(', 'unlink(', 'remove(', 'rename(']):
                return "Execute Python code with potential file operations"
            else:
                return "Execute Python code"
        
        # Generic description for other tools
        return f"Execute {tool_name} on file(s)"
    
    @staticmethod
    def should_check_permissions(tool_name: str, paths: List[Tuple[str, str]]) -> bool:
        """
        Determine if permission checking should be applied based on tool and paths.
        
        Args:
            tool_name: Name of the tool
            paths: List of detected paths
            
        Returns:
            True if permissions should be checked
        """
        # Always check permissions for tools with detected paths
        if paths:
            return True
            
        # Check permissions for potentially destructive tools even without detected paths
        destructive_tools = ["bash", "python", "writefile", "editfile"]
        return tool_name in destructive_tools
    
    @staticmethod
    def get_primary_path(paths: List[Tuple[str, str]]) -> Optional[str]:
        """
        Get the primary file path from a list of detected paths.
        
        Args:
            paths: List of (tag_name, file_path) tuples
            
        Returns:
            Primary file path or None if no paths found
        """
        if not paths:
            return None
            
        # Prefer certain tag names that are more likely to be the main target
        priority_tags = ['filepath', 'path', 'file', 'filename']
        
        for priority_tag in priority_tags:
            for tag_name, file_path in paths:
                if priority_tag in tag_name.lower():
                    return file_path
        
        # If no priority tag found, return the first path
        return paths[0][1]
    
    @staticmethod
    def get_command_hash(xml_content: str) -> str:
        """
        Generate a hash/signature for the exact command content.
        This is used for exact command matching in permissions.
        """
        import hashlib
        # Normalize the content by removing extra whitespace
        normalized = re.sub(r'\s+', ' ', xml_content.strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_command_pattern(tool_name: str, xml_content: str) -> str:
        """
        Extract a readable pattern for the command that can be used for exact matching.
        
        Args:
            tool_name: Name of the tool
            xml_content: XML content of the command
            
        Returns:
            A pattern string representing this specific command
        """
        # For bash commands, extract the actual command
        if tool_name == "bash":
            # Remove XML tags and get the actual bash command
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            # Truncate very long commands for display
            if len(clean_content) > 100:
                clean_content = clean_content[:97] + "..."
            return f"bash: {clean_content}"
        
        # For python commands, extract the python code
        elif tool_name == "python":
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            # Get first meaningful line for pattern
            first_line = clean_content.split('\n')[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            return f"python: {first_line}"
        
        # For other tools, create more specific patterns
        elif tool_name == "editfile":
            # For editfile, show the find/replace operation
            find_match = re.search(r'<find>(.*?)</find>', xml_content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', xml_content, re.DOTALL)
            if find_match and replace_match:
                find_preview = find_match.group(1)[:30] + "..." if len(find_match.group(1)) > 30 else find_match.group(1)
                replace_preview = replace_match.group(1)[:30] + "..." if len(replace_match.group(1)) > 30 else replace_match.group(1)
                return f"{tool_name}: Replace '{find_preview}' with '{replace_preview}'"
            
        # For other tools without paths, use a content-based pattern
        else:
            clean_content = re.sub(r'<[^>]*>', '', xml_content).strip()
            if len(clean_content) > 60:
                clean_content = clean_content[:57] + "..."
            return f"{tool_name}: {clean_content}"

    @staticmethod
    def analyze_tool_block(tool_name: str, xml_content: str) -> Dict:
        """
        Analyze a tool block and extract relevant information for permission checking.
        
        Args:
            tool_name: Name of the tool
            xml_content: XML content of the tool block
            
        Returns:
            Dictionary with analysis results
        """
        paths = PathDetector.extract_paths_from_xml(xml_content, tool_name)
        primary_path = PathDetector.get_primary_path(paths)
        action_description = PathDetector.get_tool_action_description(tool_name, xml_content, paths)
        needs_permission = PathDetector.should_check_permissions(tool_name, paths)
        command_hash = PathDetector.get_command_hash(xml_content)
        command_pattern = PathDetector.get_command_pattern(tool_name, xml_content)
        
        return {
            'tool_name': tool_name,
            'paths': paths,
            'primary_path': primary_path,
            'action_description': action_description,
            'needs_permission': needs_permission,
            'xml_content': xml_content,
            'command_hash': command_hash,
            'command_pattern': command_pattern,
            'has_paths': len(paths) > 0
        }