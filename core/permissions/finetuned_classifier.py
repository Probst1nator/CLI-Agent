"""
Fine-tuned EmbeddingGemma classifier using SentenceTransformers.

This classifier loads a locally fine-tuned EmbeddingGemma model and uses it
for permission classification without requiring Ollama.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Type aliases
Decision = Literal["auto-allow", "ask", "neutral"]
SafetyScore = float
EmbeddingVector = np.ndarray


class FineTunedClassifier:
    """
    Fine-tuned embedding classifier for permission classification.
    
    Uses the SentenceTransformers model directly without Ollama.
    Trained specifically for CLI-Agent permission decisions.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        safe_anchors: Optional[list[str]] = None,
        unsafe_anchors: Optional[list[str]] = None,
        threshold_allow: float = 0.02,
        threshold_deny: float = -0.02,
        device: str = "cpu"
    ):
        """
        Initialize with fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model directory
            safe_anchors: Safe concept anchors (from training)
            unsafe_anchors: Unsafe concept anchors (from training)
            threshold_allow: Score above this → auto-allow
            threshold_deny: Score below this → ask user
            device: 'cpu' or 'cuda'
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        
        # Default model path
        if model_path is None:
            model_path = str(Path.home() / ".cli-agent" / "models" / "embeddinggemma-finetuned")
        
        # Check if model exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {model_path}\n"
                "Please run train_finetuned_model.py first to create the model."
            )
        
        # Load fine-tuned model
        logger.info(f"Loading fine-tuned model from: {model_path}")
        try:
            self.model = SentenceTransformer(model_path, device=device)
            self.task_name = "Clustering"  # Must match training task
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        # Load anchors (use training defaults if not provided)
        self.safe_anchors = safe_anchors or [
            "safe", "inspection", "project files", "monitoring", "non-sensitive"
        ]
        self.unsafe_anchors = unsafe_anchors or [
            "installing", "deleting", "credentials", "network transfer",
            "sensitive data", "destructive"
        ]
        
        self.threshold_allow = threshold_allow
        self.threshold_deny = threshold_deny
        
        # Pre-compute anchor embeddings (lazy)
        self.safe_embeddings: Optional[np.ndarray] = None
        self.unsafe_embeddings: Optional[np.ndarray] = None
        self._cache: dict[str, EmbeddingVector] = {}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Loaded fine-tuned model: {param_count:,} parameters on {device}")
    
    def initialize(self) -> None:
        """
        Pre-compute anchor embeddings.
        
        This should be called once during startup to avoid lazy initialization
        latency on first classification.
        """
        if self.safe_embeddings is not None:
            logger.debug("Anchor embeddings already initialized")
            return
        
        logger.info("Computing anchor embeddings...")
        try:
            self.safe_embeddings = self.model.encode(
                self.safe_anchors,
                prompt_name=self.task_name
            )
            self.unsafe_embeddings = self.model.encode(
                self.unsafe_anchors,
                prompt_name=self.task_name
            )
            logger.info(
                f"✓ Anchor embeddings ready: {len(self.safe_anchors)} safe, "
                f"{len(self.unsafe_anchors)} unsafe"
            )
        except Exception as e:
            logger.error(f"Failed to compute anchor embeddings: {e}")
            raise
    
    def classify(self, xml_string: str) -> tuple[Decision, SafetyScore]:
        """
        Classify tool call using fine-tuned model.
        
        Args:
            xml_string: Raw XML tool call (e.g., "<bash>ls</bash>")
            
        Returns:
            Tuple of (decision, safety_score):
                - decision: "auto-allow" | "ask" | "neutral"
                - safety_score: -1.0 (unsafe) to +1.0 (safe)
        
        Examples:
            >>> classifier.classify("<bash>ls</bash>")
            ("auto-allow", 0.042)
            
            >>> classifier.classify("<bash>rm -rf /</bash>")
            ("ask", -0.078)
        """
        # Ensure initialized
        if self.safe_embeddings is None:
            self.initialize()
        
        try:
            # Get embedding for tool call
            query_emb = self.model.encode(xml_string, prompt_name=self.task_name)
            
            # Compute similarities
            safe_sims = self.model.similarity(query_emb, self.safe_embeddings).numpy()[0]
            unsafe_sims = self.model.similarity(query_emb, self.unsafe_embeddings).numpy()[0]
            
            avg_safe = float(safe_sims.mean())
            avg_unsafe = float(unsafe_sims.mean())
            
            # Safety score (same formula as training)
            safety_score = (avg_safe - avg_unsafe) / (avg_safe + avg_unsafe + 1e-10)
            
            # Apply thresholds
            if safety_score > self.threshold_allow:
                decision = "auto-allow"
            elif safety_score < self.threshold_deny:
                decision = "ask"
            else:
                decision = "neutral"
            
            logger.debug(
                f"Classification: {xml_string[:60]}... → {decision} "
                f"(score={safety_score:+.3f}, safe={avg_safe:.3f}, unsafe={avg_unsafe:.3f})"
            )
            
            return decision, safety_score
            
        except Exception as e:
            logger.error(f"Classification error for '{xml_string[:60]}...': {e}")
            # Fall back to neutral on error
            return "neutral", 0.0
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache size and anchor count
        """
        return {
            "cache_size": len(self._cache),
            "anchor_count": len(self.safe_anchors) + len(self.unsafe_anchors),
            "model_device": str(self.model.device)
        }
