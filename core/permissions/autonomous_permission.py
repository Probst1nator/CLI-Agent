"""
Autonomous permission orchestration layer.

Decision Flow (DAG):
    Tool Call (tool_name, file_path, command, xml_content)
        ↓
    ┌─────────────────────────────────────────────┐
    │ Layer 1: Static Deny Rules                 │
    │   settings_manager.check_permission()       │
    └─────────────────────────────────────────────┘
        ├─ "deny" → BLOCK (highest priority)
        ↓
    ┌─────────────────────────────────────────────┐
    │ Layer 2: Static Allow Rules                │
    │   settings_manager.check_permission()       │
    └─────────────────────────────────────────────┘
        ├─ "allow" → APPROVE
        ↓
    ┌─────────────────────────────────────────────┐
    │ Layer 3: ML Classifier (if enabled)        │
    │   embedding_classifier.classify()           │
    └─────────────────────────────────────────────┘
        ├─ "auto-allow" (score > threshold) → APPROVE
        ├─ "ask" (score < -threshold) → USER_PROMPT(suggest_deny)
        ├─ "neutral" → USER_PROMPT
        └─ error/disabled → USER_PROMPT
        ↓
    ┌─────────────────────────────────────────────┐
    │ Layer 4: User Prompt (default fallback)    │
    │   permission_prompt()                       │
    └─────────────────────────────────────────────┘
        → USER_DECISION

Performance Contract:
    - Layer 1-2: <1ms (dict/list lookup)
    - Layer 3: 10-100ms (embedding lookup/compute, cached after first call)
    - Layer 4: Blocking (waits for user input)
"""

import logging
from typing import Literal, Optional

from .embedding_classifier import (
    EmbeddingClassifier,
    SafetyScore,
    tool_call_to_xml,
    OLLAMA_AVAILABLE
)
from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)

# Type aliases
PermissionDecision = Literal["allow", "deny", "ask"]
PermissionReason = str
ConfidenceScore = float  # 0.0 (no confidence) to 1.0 (full confidence)


class AutonomousPermission:
    """
    Unified permission checker combining static rules + ML classification.

    Architecture:
        Static Layer (SettingsManager):
            - Deny list: Immediate blocking (security guardrails)
            - Allow list: Immediate approval (user-confirmed safe operations)
            - Ask list: Explicit prompting (user wants to review)

        ML Layer (EmbeddingClassifier):
            - Auto-allow: High-confidence safe operations (reduces friction)
            - Ask with warning: High-confidence unsafe operations (informed prompts)
            - Neutral: Uncertain → defer to user

    Fail-Safe Design:
        - ML disabled → static rules + user prompts
        - ML error → log warning + fall through to user prompt
        - Missing config → safe defaults (prompt for everything)

    Thread Safety:
        - Read-only after initialization (safe for concurrent access)
        - SettingsManager and EmbeddingClassifier are both read-safe
    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        ml_enabled: bool = False,
        ml_model: str = "embeddinggemma:latest",
        threshold_allow: float = 0.01,
        threshold_deny: float = -0.01,
        ollama_host: Optional[str] = None
    ):
        """
        Initialize autonomous permission checker.

        Args:
            settings_manager: Static permission rules manager
            ml_enabled: Enable ML classification layer (default: False for safety)
            ml_model: Ollama embedding model name
            threshold_allow: ML score above this → auto-allow
            threshold_deny: ML score below this → ask user (with warning)
            ollama_host: Ollama server host (None = auto-detect)

        Note:
            ML classifier is lazy-initialized on first use if ml_enabled=True.
        """
        self.settings = settings_manager
        self.ml_enabled = ml_enabled

        # ML classifier (lazy initialization)
        self.classifier: Optional[EmbeddingClassifier] = None
        self._ml_init_attempted = False

        # ML configuration (stored for lazy init)
        self._ml_config = {
            "model": ml_model,
            "threshold_allow": threshold_allow,
            "threshold_deny": threshold_deny,
            "ollama_host": ollama_host
        }

        # Statistics tracking
        self._stats = {
            "total_checks": 0,
            "static_deny": 0,
            "static_allow": 0,
            "ml_auto_allow": 0,
            "ml_ask": 0,
            "ml_neutral": 0,
            "user_prompts": 0,
            "ml_errors": 0
        }

        logger.info(
            f"Initialized AutonomousPermission: ml_enabled={ml_enabled}, "
            f"thresholds=[{threshold_deny:.3f}, {threshold_allow:.3f}]"
        )

    def _ensure_ml_initialized(self) -> bool:
        """
        Lazy-initialize ML classifier on first use.

        Flow:
            ml_enabled? → NO → return False
                ↓ YES
            already_attempted? → YES → return (classifier is not None)
                ↓ NO
            _ml_init_attempted = True
                ↓
            OLLAMA_AVAILABLE? → NO → log error, return False
                ↓ YES
            try: EmbeddingClassifier.__init__()
                ↓
            classifier.initialize() (pre-compute anchors)
                ↓
            return True

        Returns:
            True if classifier ready, False if unavailable/failed

        Side Effects:
            - Sets self.classifier on success
            - Sets self._ml_init_attempted = True
            - Logs errors on failure
        """
        if not self.ml_enabled:
            return False

        if self._ml_init_attempted:
            return self.classifier is not None

        self._ml_init_attempted = True

        if not OLLAMA_AVAILABLE:
            logger.warning(
                "ML permissions disabled: Ollama package not installed. "
                "Install with: pip install ollama"
            )
            return False

        try:
            logger.info("Initializing ML permission classifier...")
            self.classifier = EmbeddingClassifier(**self._ml_config)
            self.classifier.initialize()  # Pre-compute anchor embeddings
            logger.info("✓ ML classifier ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ML classifier: {e}")
            logger.warning("Falling back to static rules + user prompts")
            self.classifier = None
            return False

    async def check_permission(
        self,
        tool_name: str,
        file_path: Optional[str] = None,
        command: Optional[str] = None,
        xml_content: Optional[str] = None
    ) -> tuple[PermissionDecision, PermissionReason, ConfidenceScore, Optional[SafetyScore]]:
        """
        Check permission with autonomous classification.

        Flow:
            (tool_name, file_path, command, xml_content)
                ↓
            Layer 1: settings.check_permission()
                ├─ "deny" → ("deny", reason, 1.0, None)
                ↓
            Layer 2: settings.check_permission()
                ├─ "allow" → ("allow", reason, 1.0, None)
                ↓
            Layer 3: ml_enabled?
                ├─ NO → ("ask", reason, 0.0, None)
                ↓ YES
            _ensure_ml_initialized()
                ├─ failed → ("ask", reason, 0.0, None)
                ↓ success
            tool_call_to_xml() → xml_string
                ↓
            classifier.classify(xml_string) → (decision, score)
                ├─ "auto-allow" → ("allow", ml_reason, score_confidence, score)
                ├─ "ask" → ("ask", ml_warning, abs(score), score)
                └─ "neutral" → ("ask", neutral_reason, 0.0, score)

        Args:
            tool_name: Tool being executed (e.g., 'bash', 'readfile')
            file_path: File path if applicable
            command: Bash command if tool is 'bash'
            xml_content: Raw XML content (preferred for ML)

        Returns:
            Tuple of (decision, reason, confidence, ml_score):
                - decision: "allow" | "deny" | "ask"
                - reason: Human-readable explanation
                - confidence: 0.0-1.0 (how confident the decision is)
                - ml_score: Safety score from ML (-1 to +1) or None if ML not used

        Examples:
            >>> await check_permission('bash', command='sudo rm -rf /')
            ("deny", "Blocked by permission rule: bash(sudo:*)", 1.0, None)

            >>> await check_permission('readfile', file_path='./src/main.py')
            ("allow", "Permitted by rule: readfile(./src/**)", 1.0, None)

            >>> await check_permission('bash', command='ls', ml_enabled=True)
            ("allow", "ML classifier: safe (score=+0.042)", 0.85, 0.042)

            >>> await check_permission('bash', command='rm file.txt', ml_enabled=True)
            ("ask", "ML classifier: potentially unsafe (score=-0.053)", 0.73, -0.053)

        Performance:
            - Layer 1-2 (static): <1ms
            - Layer 3 (ML, first call): 50-100ms (embedding computation)
            - Layer 3 (ML, cached): 10-20ms
        """
        self._stats["total_checks"] += 1

        # Layer 1: Static deny rules (highest priority)
        decision, reason = self.settings.check_permission(tool_name, file_path, command)
        if decision == "deny":
            self._stats["static_deny"] += 1
            logger.debug(f"Permission denied by static rule: {reason}")
            return "deny", reason, 1.0, None

        # Layer 2: Static allow rules
        if decision == "allow":
            self._stats["static_allow"] += 1
            logger.debug(f"Permission allowed by static rule: {reason}")
            return "allow", reason, 1.0, None

        # Layer 3: ML classifier (if enabled and initialized)
        if self._ensure_ml_initialized() and self.classifier is not None:
            try:
                # Convert to XML format for classifier
                xml_string = tool_call_to_xml(tool_name, file_path, command, xml_content)

                # Classify with ML
                ml_decision, ml_score = self.classifier.classify(xml_string)

                if ml_decision == "auto-allow":
                    self._stats["ml_auto_allow"] += 1
                    confidence = min(abs(ml_score) * 10, 1.0)  # Scale score to confidence
                    reason = f"ML classifier: safe (score={ml_score:+.3f})"
                    logger.info(f"✓ Auto-allowed by ML: {xml_string[:60]}... (score={ml_score:+.3f})")
                    return "allow", reason, confidence, ml_score

                elif ml_decision == "ask":
                    self._stats["ml_ask"] += 1
                    confidence = min(abs(ml_score) * 10, 1.0)
                    reason = f"ML classifier: potentially unsafe (score={ml_score:+.3f})"
                    logger.info(f"⚠ ML suggests caution: {xml_string[:60]}... (score={ml_score:+.3f})")
                    return "ask", reason, confidence, ml_score

                else:  # neutral
                    self._stats["ml_neutral"] += 1
                    reason = f"ML classifier: uncertain (score={ml_score:+.3f}), requires user decision"
                    logger.debug(f"ML neutral: {xml_string[:60]}... (score={ml_score:+.3f})")
                    return "ask", reason, 0.0, ml_score

            except Exception as e:
                self._stats["ml_errors"] += 1
                logger.warning(f"ML classification error: {e}", exc_info=True)
                # Fall through to user prompt on ML failure

        # Layer 4: User prompt (default fallback)
        self._stats["user_prompts"] += 1
        return "ask", reason, 0.0, None

    def get_stats(self) -> dict:
        """
        Get permission system statistics.

        Returns:
            Dictionary with keys:
                - total_checks: Total permission checks
                - static_deny: Blocked by deny rules
                - static_allow: Approved by allow rules
                - ml_auto_allow: Auto-approved by ML (if enabled)
                - ml_ask: ML suggested asking user
                - ml_neutral: ML was uncertain
                - user_prompts: Number of user prompts shown
                - ml_errors: ML classification errors
                - ml_enabled: Whether ML is enabled
                - ml_initialized: Whether ML classifier is ready
                - cache_stats: Embedding cache stats (if ML initialized)
        """
        stats = dict(self._stats)
        stats["ml_enabled"] = self.ml_enabled
        stats["ml_initialized"] = self.classifier is not None

        if self.classifier is not None:
            stats["cache_stats"] = self.classifier.get_cache_stats()

        return stats

    def enable_ml(self) -> bool:
        """
        Enable ML classification layer.

        Flow:
            ml_enabled = True
                ↓
            _ensure_ml_initialized()
                ↓
            return success

        Returns:
            True if successfully enabled, False if initialization failed

        Side Effects:
            - Sets self.ml_enabled = True
            - Initializes classifier if not already done
        """
        self.ml_enabled = True
        success = self._ensure_ml_initialized()

        if success:
            logger.info("ML permissions enabled")
        else:
            logger.warning("ML permissions could not be enabled (check logs)")

        return success

    def disable_ml(self) -> None:
        """
        Disable ML classification layer.

        Side Effects:
            - Sets self.ml_enabled = False
            - Does NOT destroy classifier (keeps cache for re-enable)
        """
        self.ml_enabled = False
        logger.info("ML permissions disabled (classifier cache retained)")

    def get_ml_config(self) -> dict:
        """
        Get current ML configuration.

        Returns:
            Dictionary with keys:
                - enabled: Whether ML is enabled
                - initialized: Whether classifier is ready
                - model: Embedding model name
                - threshold_allow: Auto-allow threshold
                - threshold_deny: Ask-user threshold
        """
        return {
            "enabled": self.ml_enabled,
            "initialized": self.classifier is not None,
            **self._ml_config
        }
