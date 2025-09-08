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
        'code_execution': ['bash', 'python'],
        'file_management': ['bash', 'python']  # Only when they contain file operations
    }
    
    # Define sensitive keywords that should trigger automatic blocking
    SENSITIVE_KEYWORDS = {
        'sudo',           # Environment files
        '.env',           # Environment files
        'password',       # Password-related content
        'api_key',        # API keys
        'secret',         # Secrets
        'token',          # Authentication tokens
        'private_key',    # Private keys
        'id_rsa',         # SSH keys
        'id_dsa',         # SSH keys
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
        Extract file paths from XML content by finding tags containing 'path' or from command content.
        
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
        
        # For bash and python tools, also extract file paths from command content
        if tool_name in ["bash", "python"] and not paths:
            command_paths = PathDetector._extract_paths_from_command_content(xml_content, tool_name)
            paths.extend(command_paths)
        
        return paths
    
    @staticmethod
    def _extract_paths_from_command_content(xml_content: str, tool_name: str) -> List[Tuple[str, str]]:
        """
        Extract file paths from bash/python command content.
        
        Args:
            xml_content: XML content containing the command
            tool_name: Name of the tool (bash or python)
            
        Returns:
            List of (tag_name, file_path) tuples found in the command
        """
        paths = []
        
        # Remove XML tags to get the actual command content
        command_content = re.sub(r'<[^>]*>', '', xml_content).strip()
        
        if tool_name == "bash":
            paths.extend(PathDetector._extract_bash_paths(command_content))
        elif tool_name == "python":
            paths.extend(PathDetector._extract_python_paths(command_content))
            
        return paths
    
    @staticmethod
    def _extract_bash_paths(command_content: str) -> List[Tuple[str, str]]:
        """Extract file paths from bash command content - simple approach."""
        paths = []
        args = command_content.split()
        
        for arg in args:
            arg = arg.strip('\'"')  # Remove quotes
            # Simple rule: if it contains / or starts with ./ ~/ then it's a path
            if ('/' in arg or arg.startswith('./') or arg.startswith('~/')) and not arg.startswith('-'):
                abs_path = os.path.abspath(arg)
                paths.append(("bash_path", abs_path))
        
        return paths
    
    @staticmethod  
    def _extract_python_paths(command_content: str) -> List[Tuple[str, str]]:
        """Extract file paths from python command content - simple approach."""
        paths = []
        
        # Look for quoted strings that contain /
        import re
        matches = re.findall(r'["\']([^"\']*[/\\][^"\']*)["\']', command_content)
        for match in matches:
            abs_path = os.path.abspath(match)
            paths.append(("python_path", abs_path))
        
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
            # Check for line-specific editing
            line_match = re.search(r'<line>(.*?)</line>', xml_content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', xml_content, re.DOTALL)
            
            if line_match and content_match:
                line_num = line_match.group(1).strip()
                new_content = content_match.group(1)
                content_preview = new_content[:40] + "..." if len(new_content) > 40 else new_content
                return f"Edit line {line_num} → '{content_preview}'"
            else:
                # Traditional find/replace
                find_match = re.search(r'<find>(.*?)</find>', xml_content, re.DOTALL)
                replace_match = re.search(r'<replace>(.*?)</replace>', xml_content, re.DOTALL)
                
                find_preview = find_match.group(1)[:40] + "..." if find_match and len(find_match.group(1)) > 40 else find_match.group(1) if find_match else "?"
                replace_preview = replace_match.group(1)[:40] + "..." if replace_match and len(replace_match.group(1)) > 40 else replace_match.group(1) if replace_match else "?"
                
                return f"Replace '{find_preview}' with '{replace_preview}'"
            
        elif tool_name == "writefile":
            return "Write/overwrite file content"
            
        elif tool_name == "readfile":
            # Check for line-specific reading
            lines_match = re.search(r'<lines>(.*?)</lines>', xml_content, re.DOTALL)
            if lines_match:
                line_spec = lines_match.group(1).strip()
                return f"Read lines {line_spec}"
            else:
                return "Read file content"
        
        elif tool_name == "insertline":
            content_match = re.search(r'<content>(.*?)</content>', xml_content, re.DOTALL)
            after_match = re.search(r'<after>(.*?)</after>', xml_content, re.DOTALL)
            before_match = re.search(r'<before>(.*?)</before>', xml_content, re.DOTALL)
            
            content_preview = content_match.group(1)[:40] + "..." if content_match and len(content_match.group(1)) > 40 else content_match.group(1) if content_match else "?"
            
            if after_match:
                return f"Insert after line {after_match.group(1).strip()} → '{content_preview}'"
            elif before_match:
                return f"Insert before line {before_match.group(1).strip()} → '{content_preview}'"
            else:
                return f"Insert at end → '{content_preview}'"
        
        elif tool_name == "deletelines":
            line_match = re.search(r'<line>(.*?)</line>', xml_content, re.DOTALL)
            start_match = re.search(r'<start>(.*?)</start>', xml_content, re.DOTALL)
            end_match = re.search(r'<end>(.*?)</end>', xml_content, re.DOTALL)
            
            if line_match:
                return f"Delete line {line_match.group(1).strip()}"
            elif start_match and end_match:
                return f"Delete lines {start_match.group(1).strip()}-{end_match.group(1).strip()}"
            else:
                return "Delete lines"
            
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
    def contains_sensitive_keywords(content: str) -> Tuple[bool, List[str]]:
        """
        Check if content contains any sensitive keywords that should block execution.
        
        Args:
            content: The content to check (file paths, command text, etc.)
            
        Returns:
            Tuple of (has_sensitive_content, list_of_found_keywords)
        """
        content_lower = content.lower()
        found_keywords = []
        
        for keyword in PathDetector.SENSITIVE_KEYWORDS:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    @staticmethod
    def check_sensitive_content_in_xml(xml_content: str) -> Tuple[bool, List[str]]:
        """
        Check XML content for sensitive keywords in paths, commands, or content.
        
        Args:
            xml_content: XML content to analyze
            
        Returns:
            Tuple of (has_sensitive_content, list_of_found_keywords)
        """
        all_found_keywords = []
        
        # Check the entire XML content
        has_sensitive, keywords = PathDetector.contains_sensitive_keywords(xml_content)
        if has_sensitive:
            all_found_keywords.extend(keywords)
        
        return len(all_found_keywords) > 0, all_found_keywords
    
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