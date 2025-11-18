"""
Agent Discovery System for Agent SaaS Infrastructure

Discovers available agents from the ./agents/ directory and provides
metadata about each agent (tools, description, requirements).
"""

import importlib.util
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Definition of a discovered agent"""
    name: str                    # Internal name (e.g., "minimal_agent")
    display_name: str            # Human-readable name (e.g., "Minimal Agent")
    description: str             # Short description
    tools: list[str]             # List of available tool names
    path: str                    # Absolute path to agent directory
    docker_image: str            # Docker image name (e.g., "cli-agent-minimal:latest")
    icon: str                    # Icon/emoji for UI
    category: str                # Category (e.g., "development", "analysis")
    tool_count: int              # Number of tools
    has_write_access: bool       # Whether agent has write tools (bash, writefile, etc.)


class AgentDiscovery:
    """
    Discovers and enumerates available agents from the filesystem.

    Scans ./agents/ directory for valid agent subdirectories and extracts
    metadata from each agent's structure.
    """

    def __init__(self, agents_base_path: Path = None):
        """
        Initialize agent discovery.

        Args:
            agents_base_path: Path to agents directory (defaults to ./agents/)
        """
        if agents_base_path is None:
            # Default to project root's agents directory
            agents_base_path = Path(__file__).parent.parent / "agents"

        self.agents_base_path = agents_base_path
        self.discovered_agents: dict[str, AgentDefinition] = {}
        self._scan_agents()

    def _scan_agents(self):
        """Scan agents directory and discover available agents"""
        if not self.agents_base_path.exists():
            logger.error(f"Agents directory not found: {self.agents_base_path}")
            return

        logger.info(f"Scanning for agents in: {self.agents_base_path}")

        for agent_dir in self.agents_base_path.iterdir():
            if not agent_dir.is_dir():
                continue

            # Skip hidden directories and __pycache__
            if agent_dir.name.startswith('.') or agent_dir.name.startswith('__'):
                continue

            # Check if it's a valid agent (has main.py)
            main_py = agent_dir / "main.py"
            if not main_py.exists():
                logger.debug(f"Skipping {agent_dir.name}: no main.py found")
                continue

            try:
                agent_def = self._parse_agent(agent_dir)
                if agent_def:
                    self.discovered_agents[agent_def.name] = agent_def
                    logger.info(f"Discovered agent: {agent_def.display_name} ({agent_def.tool_count} tools)")
            except Exception as e:
                logger.error(f"Failed to parse agent {agent_dir.name}: {e}")

    def _parse_agent(self, agent_dir: Path) -> Optional[AgentDefinition]:
        """
        Parse an agent directory and extract metadata.

        Args:
            agent_dir: Path to agent directory

        Returns:
            AgentDefinition or None if parsing fails
        """
        agent_name = agent_dir.name

        # Read description from README.md or CLAUDE.md
        description = self._extract_description(agent_dir)

        # Discover tools
        tools = self._discover_tools(agent_dir)

        # Determine if agent has write access
        write_tools = {'writefile', 'editfile', 'bash', 'git'}
        has_write_access = bool(write_tools.intersection(set(tools)))

        # Generate display name (convert snake_case to Title Case)
        display_name = agent_name.replace('_', ' ').title()

        # Determine category based on tools
        category = self._determine_category(tools)

        # Docker image name
        docker_image = f"cli-agent-{agent_name}:latest"

        # Icon based on category
        icon = self._get_icon(category, agent_name)

        return AgentDefinition(
            name=agent_name,
            display_name=display_name,
            description=description,
            tools=tools,
            path=str(agent_dir.absolute()),
            docker_image=docker_image,
            icon=icon,
            category=category,
            tool_count=len(tools),
            has_write_access=has_write_access
        )

    def _extract_description(self, agent_dir: Path) -> str:
        """Extract description from README.md or CLAUDE.md"""
        # Try README.md first
        readme = agent_dir / "README.md"
        if readme.exists():
            try:
                content = readme.read_text(encoding='utf-8')
                # Extract first paragraph or first line after title
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                for line in lines:
                    # Skip markdown headers
                    if line.startswith('#'):
                        continue
                    # Return first non-header line
                    if len(line) > 20:  # Reasonable description length
                        return line[:200]  # Max 200 chars
            except Exception as e:
                logger.debug(f"Failed to read README.md: {e}")

        # Try CLAUDE.md
        claude_md = agent_dir / "CLAUDE.md"
        if claude_md.exists():
            try:
                content = claude_md.read_text(encoding='utf-8')
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if len(line) > 20:
                        return line[:200]
            except Exception as e:
                logger.debug(f"Failed to read CLAUDE.md: {e}")

        # Try extracting from main.py docstring
        main_py = agent_dir / "main.py"
        if main_py.exists():
            try:
                content = main_py.read_text(encoding='utf-8')
                # Look for docstring after imports
                match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if match:
                    docstring = match.group(1).strip()
                    # Get first line
                    first_line = docstring.split('\n')[0].strip()
                    if len(first_line) > 20:
                        return first_line[:200]
            except Exception as e:
                logger.debug(f"Failed to extract docstring: {e}")

        # Default description
        return f"A specialized agent with {agent_dir.name} capabilities"

    def _discover_tools(self, agent_dir: Path) -> list[str]:
        """Discover tools by scanning tools/ directory"""
        tools_dir = agent_dir / "tools"
        if not tools_dir.exists():
            return []

        tools = []
        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name.startswith('_'):
                continue

            tool_name = tool_file.stem

            # Verify it's a valid tool by checking for get_delim method
            try:
                spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for class with same name as module
                    if hasattr(module, tool_name):
                        tool_class = getattr(module, tool_name)
                        if hasattr(tool_class, 'get_delim'):
                            tools.append(tool_name)
            except Exception as e:
                logger.debug(f"Failed to load tool {tool_name}: {e}")

        return sorted(tools)

    def _determine_category(self, tools: list[str]) -> str:
        """Determine agent category based on available tools"""
        tool_set = set(tools)

        # Development tools
        dev_tools = {'bash', 'git', 'github', 'writefile', 'editfile'}
        if tool_set.intersection(dev_tools):
            return "development"

        # Analysis tools (read-only)
        if tool_set.intersection({'readfile', 'grepsearch'}) and not tool_set.intersection(dev_tools):
            return "analysis"

        # Cognitive tools
        if tool_set.intersection({'think', 'todos'}):
            return "cognitive"

        return "general"

    def _get_icon(self, category: str, agent_name: str) -> str:
        """Get icon based on category or agent name"""
        category_icons = {
            "development": "🛠️",
            "analysis": "🔍",
            "cognitive": "🧠",
            "general": "🤖"
        }

        # Special cases
        if "minimal" in agent_name.lower():
            return "⚡"
        if "developer" in agent_name.lower():
            return "👨‍💻"

        return category_icons.get(category, "🤖")

    def get_agent(self, agent_name: str) -> Optional[AgentDefinition]:
        """
        Get agent definition by name.

        Args:
            agent_name: Agent name (e.g., "minimal_agent")

        Returns:
            AgentDefinition or None if not found
        """
        return self.discovered_agents.get(agent_name)

    def list_agents(self) -> list[AgentDefinition]:
        """
        Get list of all discovered agents.

        Returns:
            List of AgentDefinition objects
        """
        return list(self.discovered_agents.values())

    def list_agents_by_category(self) -> dict[str, list[AgentDefinition]]:
        """
        Get agents grouped by category.

        Returns:
            Dictionary mapping category -> list of agents
        """
        by_category = {}
        for agent in self.discovered_agents.values():
            if agent.category not in by_category:
                by_category[agent.category] = []
            by_category[agent.category].append(agent)
        return by_category

    def to_json(self) -> str:
        """Export all agents as JSON"""
        agents_list = [asdict(agent) for agent in self.list_agents()]
        return json.dumps(agents_list, indent=2)

    def validate_agent(self, agent_name: str) -> tuple[bool, str]:
        """
        Validate that an agent exists and is properly configured.

        Args:
            agent_name: Agent name to validate

        Returns:
            (is_valid, error_message)
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return False, f"Agent '{agent_name}' not found"

        agent_path = Path(agent.path)

        # Check main.py exists
        if not (agent_path / "main.py").exists():
            return False, f"Agent '{agent_name}' missing main.py"

        # Check tools directory exists
        if not (agent_path / "tools").exists():
            return False, f"Agent '{agent_name}' missing tools directory"

        # Check at least one tool
        if agent.tool_count == 0:
            return False, f"Agent '{agent_name}' has no tools"

        return True, ""


# Global instance for easy access
_discovery_instance: Optional[AgentDiscovery] = None


def get_discovery() -> AgentDiscovery:
    """Get global AgentDiscovery instance (singleton pattern)"""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = AgentDiscovery()
    return _discovery_instance


def refresh_discovery():
    """Refresh agent discovery (rescan filesystem)"""
    global _discovery_instance
    _discovery_instance = AgentDiscovery()
    return _discovery_instance


if __name__ == "__main__":
    # Test discovery
    logging.basicConfig(level=logging.INFO)

    discovery = AgentDiscovery()

    print("=== Discovered Agents ===")
    for agent in discovery.list_agents():
        print(f"\n{agent.icon} {agent.display_name}")
        print(f"  Name: {agent.name}")
        print(f"  Category: {agent.category}")
        print(f"  Tools ({agent.tool_count}): {', '.join(agent.tools)}")
        print(f"  Write Access: {agent.has_write_access}")
        print(f"  Description: {agent.description}")
        print(f"  Docker Image: {agent.docker_image}")

    print("\n=== By Category ===")
    by_category = discovery.list_agents_by_category()
    for category, agents in by_category.items():
        print(f"\n{category.title()}:")
        for agent in agents:
            print(f"  - {agent.display_name}")

    print("\n=== JSON Export ===")
    print(discovery.to_json())
