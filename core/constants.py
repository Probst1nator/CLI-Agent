"""
Central constants for CLI-Agent tag definitions.

This module centralizes tag definitions that are used across multiple components
to ensure consistency and avoid duplication.
"""

# Tags that should NOT interrupt the streaming response
# These tags contain free-form content (thinking, planning) not tool calls
NON_INTERRUPTING_TAGS = {'think', 'todos'}

# Tags that should NOT be validated as structured XML
# These tags contain free-form text, not XML structure
NON_VALIDATING_TAGS = {'think', 'todos'}

# Tags that have priority in streaming output
# These are rendered/displayed with priority in the stream painter
PRIORITY_TAGS = {'python', 'bash', 'think', 'todos', 'editfile', 'writefile', 'readfile', 'viewfiles', 'tool'}

# Tags that are non-executable cognitive tools (return None)
# These help the model reason but don't produce external outputs
COGNITIVE_TOOLS = {'think', 'todos'}

# Tools that should NEVER require permission checks
# These are primitive/structural tools that are always safe to execute
PERMISSION_EXEMPT_TOOLS = {'response', 'visualresponse', 'think', 'todos', 'searchweb', 'grepsearch'}
