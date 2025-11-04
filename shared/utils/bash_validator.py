"""
Bash command validation and sanitization for CLI-Agent.

Provides security checks and audit logging for bash tool commands.
Identifies high-risk commands and patterns that could be dangerous.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# High-risk commands that should always be flagged
DANGEROUS_COMMANDS = {
    'rm', 'dd', 'mkfs', 'fdisk', 'parted',
    'shred', 'wipefs', ':(){:|:&};:',  # Fork bomb
    'curl | sh', 'wget | sh', 'curl | bash', 'wget | bash',
}

# Commands requiring elevated privileges
PRIVILEGE_COMMANDS = {
    'sudo', 'su', 'doas', 'pkexec'
}

# System modification commands
SYSTEM_MODIFICATION = {
    'chmod', 'chown', 'chgrp', 'useradd', 'userdel', 'groupadd', 'groupdel',
    'passwd', 'systemctl', 'service', 'reboot', 'shutdown', 'halt', 'poweroff'
}

# Network commands that could exfiltrate data
NETWORK_COMMANDS = {
    'nc', 'netcat', 'ncat', 'socat', 'telnet', 'ftp', 'tftp', 'scp', 'rsync'
}

# Command injection patterns
INJECTION_PATTERNS = [
    r'`.*`',  # Backticks
    r'\$\(.*\)',  # Command substitution
    r'&&',  # Command chaining
    r'\|\|',  # Command chaining
    r';',  # Command separator
    r'\|',  # Pipe
    r'>',  # Redirect
    r'<',  # Redirect
    r'>>',  # Append redirect
]


def extract_command_name(command: str) -> str:
    """
    Extract the base command name from a bash command.

    Args:
        command: Full command string

    Returns:
        Base command name (first word)
    """
    # Remove leading/trailing whitespace
    command = command.strip()

    # Split on whitespace and get first word
    parts = command.split()
    if not parts:
        return ""

    # Handle sudo/doas prefix
    if parts[0] in PRIVILEGE_COMMANDS and len(parts) > 1:
        return parts[1]

    return parts[0]


def analyze_command_risk(command: str) -> Tuple[str, str, list]:
    """
    Analyze command for security risks.

    Args:
        command: Bash command to analyze

    Returns:
        Tuple of (risk_level, reason, warnings)
        risk_level: 'critical', 'high', 'medium', 'low'
        reason: Description of risk
        warnings: List of specific warnings
    """
    warnings = []
    cmd_lower = command.lower()

    # Extract base command
    base_cmd = extract_command_name(command)

    # Check for fork bomb
    if ':(){:|:&};:' in command or ':(){ :|:& };:' in command:
        return 'critical', 'Fork bomb detected', ['FORK BOMB PATTERN']

    # Check for dangerous commands
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            warnings.append(f"Dangerous command: {dangerous}")

    # Check for privilege escalation
    for priv_cmd in PRIVILEGE_COMMANDS:
        if priv_cmd in cmd_lower:
            warnings.append(f"Privilege escalation: {priv_cmd}")

    # Check for system modification
    for sys_cmd in SYSTEM_MODIFICATION:
        if base_cmd == sys_cmd or f' {sys_cmd} ' in cmd_lower:
            warnings.append(f"System modification: {sys_cmd}")

    # Check for network commands
    for net_cmd in NETWORK_COMMANDS:
        if base_cmd == net_cmd or f' {net_cmd} ' in cmd_lower:
            warnings.append(f"Network command: {net_cmd}")

    # Check for command injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, command):
            warnings.append(f"Command injection pattern: {pattern}")

    # Check for pipe-to-shell patterns (very dangerous)
    if re.search(r'(curl|wget|fetch)\s+.*\s*\|\s*(sh|bash|zsh|fish)', cmd_lower):
        return 'critical', 'Pipe-to-shell detected (curl|wget | sh)', warnings

    # Check for rm with dangerous flags
    if 'rm' in cmd_lower:
        if '-rf' in cmd_lower or '-fr' in cmd_lower:
            if '/' in command or '*' in command:
                return 'critical', 'Recursive rm with wildcards/root path', warnings

    # Determine risk level based on warnings
    if any('FORK BOMB' in w or 'Pipe-to-shell' in w for w in warnings):
        return 'critical', 'Critical security risk detected', warnings
    elif len(warnings) >= 3:
        return 'high', 'Multiple security concerns', warnings
    elif warnings:
        return 'medium', 'Security concerns detected', warnings
    else:
        return 'low', 'No significant security concerns', []


def log_bash_command(command: str, user_id: Optional[str] = None, risk_level: str = 'low'):
    """
    Log bash command to audit log.

    Args:
        command: Command being executed
        user_id: User ID (UUID) if available
        risk_level: Risk level from analyze_command_risk
    """
    audit_dir = Path.home() / ".cli-agent" / "audit_logs" / "bash_commands"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Create log entry
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id or 'unknown',
        'command': command,
        'risk_level': risk_level
    }

    # Append to today's log file
    log_file = audit_dir / f"bash_audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    try:
        import json
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        logger.debug(f"Logged bash command (risk: {risk_level})")
    except Exception as e:
        logger.error(f"Failed to log bash command: {e}")


def validate_command(command: str, user_id: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate bash command and return whether it should be allowed.

    This is a defense-in-depth check. Permission system should
    be the primary enforcement mechanism.

    Args:
        command: Command to validate
        user_id: User ID (UUID) if available

    Returns:
        Tuple of (is_allowed, reason)
    """
    # Analyze risk
    risk_level, reason, warnings = analyze_command_risk(command)

    # Log all commands
    log_bash_command(command, user_id, risk_level)

    # Block critical risk commands
    if risk_level == 'critical':
        logger.warning(f"BLOCKED CRITICAL COMMAND: {command[:100]}")
        return False, f"Command blocked: {reason}"

    # Warn about high-risk commands but allow (permission system handles this)
    if risk_level == 'high':
        logger.warning(f"HIGH RISK COMMAND: {command[:100]} - {warnings}")

    return True, "Command validation passed"
