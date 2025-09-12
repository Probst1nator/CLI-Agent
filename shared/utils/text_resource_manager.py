"""
TextResourceManager - Global singleton for managing large text content

This module provides a unified system for handling large text content from files,
process outputs, and other sources. It allows tools to store content once and
provide agents with intelligent search and navigation capabilities.

Key Features:
- Global singleton pattern for consistent resource management
- Unique resource IDs for content identification  
- Memory-efficient storage with automatic cleanup
- Metadata tracking (line counts, character counts, source info)
- Thread-safe operations for concurrent access
"""
import threading
import time
import re
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetadata:
    """Metadata for a text resource."""
    resource_id: str
    source: str  # file path, process ID, etc.
    source_type: str  # 'file', 'process', 'buffer', etc.
    total_lines: int
    total_chars: int
    created_at: float  # timestamp
    last_accessed: float  # timestamp for cleanup
    encoding: str = 'utf-8'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TextResource:
    """Container for large text content with metadata and search capabilities."""
    
    def __init__(self, content: str, metadata: ResourceMetadata):
        self.content = content
        self.metadata = metadata
        self._lines_cache: Optional[List[str]] = None
        self._lock = threading.RLock()
    
    @property
    def lines(self) -> List[str]:
        """Get content as list of lines (cached)."""
        if self._lines_cache is None:
            with self._lock:
                if self._lines_cache is None:  # Double-check locking
                    self._lines_cache = self.content.splitlines()
        return self._lines_cache
    
    def search_regex(self, pattern: str, context_lines: int = 0, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for regex pattern in content.
        
        Args:
            pattern: Regular expression pattern to search for
            context_lines: Number of lines before/after each match to include
            max_results: Maximum number of matches to return
            
        Returns:
            List of match dictionaries with line numbers, content, and context
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return [{"error": f"Invalid regex pattern '{pattern}': {str(e)}"}]
        
        matches = []
        lines = self.lines
        total_lines = len(lines)
        
        for line_num, line in enumerate(lines, 1):
            if regex.search(line):
                # Calculate context window
                start_line = max(1, line_num - context_lines)
                end_line = min(total_lines, line_num + context_lines)
                
                context = []
                for ctx_line_num in range(start_line, end_line + 1):
                    ctx_line = lines[ctx_line_num - 1]  # Convert to 0-based index
                    is_match = (ctx_line_num == line_num)
                    context.append({
                        "line_number": ctx_line_num,
                        "content": ctx_line,
                        "is_match": is_match
                    })
                
                matches.append({
                    "line_number": line_num,
                    "matched_line": line,
                    "context": context,
                    "match_preview": line[:100] + ("..." if len(line) > 100 else "")
                })
                
                if len(matches) >= max_results:
                    break
        
        return matches
    
    def get_line_range(self, start_line: int, end_line: Optional[int] = None) -> Dict[str, Any]:
        """
        Get specific line range from content.
        
        Args:
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based), or None for single line
            
        Returns:
            Dictionary with line range content and metadata
        """
        lines = self.lines
        total_lines = len(lines)
        
        if start_line < 1 or start_line > total_lines:
            return {"error": f"Start line {start_line} is out of range (1-{total_lines})"}
        
        if end_line is None:
            end_line = start_line
        
        if end_line < start_line:
            return {"error": f"End line {end_line} cannot be before start line {start_line}"}
        
        if end_line > total_lines:
            end_line = total_lines
        
        # Extract lines (convert to 0-based indexing)
        selected_lines = []
        for line_num in range(start_line, end_line + 1):
            selected_lines.append({
                "line_number": line_num,
                "content": lines[line_num - 1]
            })
        
        return {
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "lines": selected_lines
        }
    
    def update_access_time(self):
        """Update last accessed timestamp."""
        self.metadata.last_accessed = time.time()


class TextResourceManager:
    """
    Global singleton for managing large text resources.
    
    Provides thread-safe storage, retrieval, and cleanup of large text content
    with intelligent search and navigation capabilities.
    """
    
    _instance: Optional['TextResourceManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Private constructor - use _get_manager() instead."""
        if TextResourceManager._instance is not None:
            raise Exception("TextResourceManager is a singleton! Use _get_manager() to access it.")
        self._initialize()
    
    @staticmethod
    def _get_manager() -> 'TextResourceManager':
        """
        Get the singleton instance, creating it if necessary.
        This follows the same lazy initialization pattern used by bash.py and other tools.
        """
        if TextResourceManager._instance is None:
            with TextResourceManager._lock:
                if TextResourceManager._instance is None:
                    TextResourceManager._instance = TextResourceManager.__new__(TextResourceManager)
                    TextResourceManager._instance._initialize()
        return TextResourceManager._instance
    
    def _initialize(self):
        """Initialize the manager (called once)."""
        self._resources: Dict[str, TextResource] = {}
        self._resource_lock = threading.RLock()
        self._cleanup_interval = 3600  # 1 hour
        self._max_age = 86400  # 24 hours
        self._max_resources = 100  # Maximum cached resources
        
        # Safe logging - by the time tools are used, logging will be configured
        try:
            logger.info("TextResourceManager initialized")
        except (ValueError, OSError):
            # Logging system may not be ready during early initialization
            pass
    
    def create_resource(
        self, 
        content: str, 
        source: str, 
        source_type: str = 'unknown',
        resource_id: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> str:
        """
        Create a new text resource.
        
        Args:
            content: The text content to store
            source: Source identifier (file path, process ID, etc.)
            source_type: Type of source ('file', 'process', 'buffer', etc.)
            resource_id: Optional custom resource ID, auto-generated if not provided
            encoding: Text encoding used
            
        Returns:
            The resource ID for the created resource
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        # Generate resource ID if not provided
        if resource_id is None:
            # Create deterministic ID based on source and content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            resource_id = f"{source_type}_{content_hash}"
        
        # Create metadata
        metadata = ResourceMetadata(
            resource_id=resource_id,
            source=source,
            source_type=source_type,
            total_lines=len(content.splitlines()),
            total_chars=len(content),
            created_at=time.time(),
            last_accessed=time.time(),
            encoding=encoding
        )
        
        # Create and store resource
        resource = TextResource(content, metadata)
        
        with self._resource_lock:
            # Check if we need to clean up old resources
            if len(self._resources) >= self._max_resources:
                self._cleanup_old_resources()
            
            self._resources[resource_id] = resource
            
        try:
            logger.debug(f"Created text resource {resource_id} ({metadata.total_lines} lines, {metadata.total_chars} chars)")
        except (ValueError, OSError):
            pass
        return resource_id
    
    def get_resource(self, resource_id: str) -> Optional[TextResource]:
        """Get a text resource by ID."""
        with self._resource_lock:
            resource = self._resources.get(resource_id)
            if resource:
                resource.update_access_time()
            return resource
    
    def get_metadata(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a resource without loading content."""
        resource = self.get_resource(resource_id)
        return resource.metadata.to_dict() if resource else None
    
    def search_resource(
        self, 
        resource_id: str, 
        pattern: str, 
        context_lines: int = 0, 
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Search for pattern in a specific resource.
        
        Args:
            resource_id: ID of the resource to search
            pattern: Regular expression pattern
            context_lines: Lines of context around matches
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results or error message
        """
        resource = self.get_resource(resource_id)
        if not resource:
            return {"error": f"Resource '{resource_id}' not found"}
        
        try:
            matches = resource.search_regex(pattern, context_lines, max_results)
            return {
                "resource_id": resource_id,
                "pattern": pattern,
                "total_matches": len(matches),
                "matches": matches,
                "context_lines": context_lines,
                "source": resource.metadata.source,
                "source_type": resource.metadata.source_type
            }
        except Exception as e:
            try:
                logger.error(f"Search error in resource {resource_id}: {e}")
            except (ValueError, OSError):
                pass
            return {"error": f"Search failed: {str(e)}"}
    
    def get_line_range(
        self, 
        resource_id: str, 
        start_line: int, 
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get specific line range from a resource."""
        resource = self.get_resource(resource_id)
        if not resource:
            return {"error": f"Resource '{resource_id}' not found"}
        
        try:
            result = resource.get_line_range(start_line, end_line)
            result["resource_id"] = resource_id
            result["source"] = resource.metadata.source
            return result
        except Exception as e:
            try:
                logger.error(f"Line range error in resource {resource_id}: {e}")
            except (ValueError, OSError):
                pass
            return {"error": f"Failed to get line range: {str(e)}"}
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources."""
        with self._resource_lock:
            resources = []
            for resource_id, resource in self._resources.items():
                resources.append({
                    "resource_id": resource_id,
                    "source": resource.metadata.source,
                    "source_type": resource.metadata.source_type,
                    "total_lines": resource.metadata.total_lines,
                    "total_chars": resource.metadata.total_chars,
                    "created_at": resource.metadata.created_at,
                    "last_accessed": resource.metadata.last_accessed
                })
            return sorted(resources, key=lambda x: x["last_accessed"], reverse=True)
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource from the manager."""
        with self._resource_lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
                try:
                    logger.debug(f"Removed resource {resource_id}")
                except (ValueError, OSError):
                    pass
                return True
            return False
    
    def clear_all_resources(self):
        """Clear all resources (useful for testing)."""
        with self._resource_lock:
            self._resources.clear()
            try:
                logger.debug("Cleared all text resources")
            except (ValueError, OSError):
                pass
    
    def _cleanup_old_resources(self):
        """Remove old or least recently used resources."""
        current_time = time.time()
        resources_to_remove = []
        
        # Find resources to remove (old or over limit)
        for resource_id, resource in self._resources.items():
            age = current_time - resource.metadata.last_accessed
            if age > self._max_age:
                resources_to_remove.append(resource_id)
        
        # If still over limit, remove least recently used
        if len(self._resources) - len(resources_to_remove) >= self._max_resources:
            sorted_resources = sorted(
                self._resources.items(),
                key=lambda x: x[1].metadata.last_accessed
            )
            excess_count = len(self._resources) - len(resources_to_remove) - self._max_resources + 10
            for i in range(excess_count):
                if i < len(sorted_resources):
                    resources_to_remove.append(sorted_resources[i][0])
        
        # Remove selected resources
        for resource_id in resources_to_remove:
            if resource_id in self._resources:
                del self._resources[resource_id]
        
        if resources_to_remove:
            try:
                logger.debug(f"Cleaned up {len(resources_to_remove)} old text resources")
            except (ValueError, OSError):
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._resource_lock:
            total_lines = sum(r.metadata.total_lines for r in self._resources.values())
            total_chars = sum(r.metadata.total_chars for r in self._resources.values())
            
            return {
                "total_resources": len(self._resources),
                "total_lines": total_lines,
                "total_chars": total_chars,
                "max_resources": self._max_resources,
                "max_age_seconds": self._max_age
            }


# Use TextResourceManager._get_manager() to access the singleton instance