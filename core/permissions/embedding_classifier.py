"""
Embedding-based autonomous permission classifier.

Decision Flow (DAG):
    XML Tool Call
        ↓
    get_embedding(xml_string)
        ↓
    [safe_embeddings, unsafe_embeddings] ← initialize()
        ↓
    compute_similarity(embedding, safe_group)
    compute_similarity(embedding, unsafe_group)
        ↓
    safety_score = (avg_safe - avg_unsafe) / (avg_safe + avg_unsafe)
        ↓
    classify() → (decision, score)
        ├─ score > threshold_allow → "auto-allow"
        ├─ score < threshold_deny → "ask"
        └─ else → "neutral"
"""

import logging
import os
from typing import Literal, Optional

import numpy as np

# Conditional import for Ollama (graceful degradation)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Conditional import for fine-tuned classifier
try:
    from .finetuned_classifier import FineTunedClassifier as _FineTunedClassifier
    FINETUNED_AVAILABLE = True
except ImportError:
    FINETUNED_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for clarity
SafetyScore = float  # Range: -1.0 (unsafe) to +1.0 (safe)
Decision = Literal["auto-allow", "ask", "neutral"]
EmbeddingVector = np.ndarray  # Shape: (embedding_dim,)


class EmbeddingClassifier:
    """
    1D embedding-based safety classifier using cosine similarity.

    Architecture:
        - Anchor-based: Compares tool calls to 18 safe + 29 unsafe reference concepts
        - Raw XML: Uses actual LLM-generated XML (no transformation)
        - 1D scoring: Single safety score from similarity differential
        - Threshold-based: Configurable decision boundaries

    Decision Logic:
        score > +0.01 → AUTO-ALLOW (safe)
        score < -0.01 → ASK USER (unsafe)
        -0.01 ≤ score ≤ +0.01 → NEUTRAL (use explicit rules)

    Performance (from notebook validation):
        - Accuracy: 86.2% at threshold ±0.01
        - Auto-allow rate: 57% of safe operations
        - False positive rate: ~5% (safe → ask)

    Thread Safety:
        - Read-only after initialization
        - Cache is dict (not thread-safe for writes, but safe for reads after init)
    """

    # Anchor concepts: Diverse, high-variance examples (ONE per semantic category)
    SAFE_CONCEPTS: list[str] = [
        # Bash: Information queries (one per category)
        "<bash>ls</bash>",                    # File listing
        "<bash>pwd</bash>",                   # Location query
        "<bash>cat file.txt</bash>",          # File reading
        "<bash>grep pattern file</bash>",     # Text search
        "<bash>git status</bash>",            # Version control status
        "<bash>ps aux</bash>",                # Process listing
        "<bash>df -h</bash>",                 # Disk info

        # File tools: Read-only
        "<readfile><filepath>./file.py</filepath></readfile>",
        "<readdirectory><path>.</path></readdirectory>",

        # Search tools
        "<grepsearch><path>./</path></grepsearch>",
        "<searchfile><path>./src</path></searchfile>",
        "<searchconversations></searchconversations>",

        # Internal/harmless tools
        "<think></think>",
        "<todos></todos>",
        "<visualresponse></visualresponse>",
        "<help></help>",
        "<viewpermissions></viewpermissions>",

        # Custom safe operations
        "read XXX.log",
    ]

    UNSAFE_CONCEPTS: list[str] = [
        # Bash: Destructive operations (one per category)
        "<bash>rm file.txt</bash>",           # File deletion
        "<bash>rm -rf /</bash>",              # Dangerous deletion
        "<bash>chmod +x script.sh</bash>",    # Permission change
        "<bash>pip install package</bash>",   # Package install
        "<bash>sudo apt update</bash>",       # Privileged operation
        "<bash>git push</bash>",              # Version control write
        "<bash>docker run image</bash>",      # Container execution
        "<bash>kill -9 pid</bash>",           # Process termination
        "<bash>useradd user</bash>",          # User management

        # File tools: Write operations (one per type)
        "<writefile><filepath>./file.py</filepath></writefile>",
        "<editfile><filepath>./file.py</filepath></editfile>",
        "<insertline><filepath>./file.py</filepath></insertline>",
        "<deletelines><filepath>./file.py</filepath></deletelines>",
        "<appendfile><filepath>./file.py</filepath></appendfile>",
        "<readfile><filepath>.env</filepath></readfile>",

        # System control tools
        "<computercontrol></computercontrol>",
        "<taskmanager></taskmanager>",
        "<bug_report></bug_report>",

        # Abstract anchors for semantic coverage
        "unsafe",
        "destructive",
        "writing",
        "deleting",
        "modifying",
        "installing",
        "executing",
        "dangerous",

        # Network/sensitive patterns
        "curl",
        "wget",
        ".env",
    ]

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        threshold_allow: float = 0.01,
        threshold_deny: float = -0.01,
        ollama_host: Optional[str] = None,
        use_finetuned: bool = True
    ):
        """
        Initialize classifier with configurable thresholds.

        Args:
            model: Ollama embedding model name (used if fine-tuned not available)
            threshold_allow: Score above this → auto-allow (default: 0.01)
            threshold_deny: Score below this → ask user (default: -0.01)
            ollama_host: Ollama server host (default: from OLLAMA_HOST env or localhost)
            use_finetuned: Prefer fine-tuned local model over Ollama (default: True)

        Raises:
            RuntimeError: If neither fine-tuned model nor Ollama is available
        """
        self.threshold_allow = threshold_allow
        self.threshold_deny = threshold_deny
        self._use_finetuned = False
        self._impl = None
        
        # Try to use fine-tuned model first
        if use_finetuned and FINETUNED_AVAILABLE:
            try:
                logger.info("Attempting to use fine-tuned local model...")
                self._impl = _FineTunedClassifier(
                    threshold_allow=threshold_allow,
                    threshold_deny=threshold_deny
                )
                self._use_finetuned = True
                logger.info("✓ Using fine-tuned local model for permission classification")
                return
            except FileNotFoundError as e:
                logger.warning(f"Fine-tuned model not found: {e}")
                logger.info("Falling back to Ollama...")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}")
                logger.info("Falling back to Ollama...")
        
        # Fall back to Ollama
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "Neither fine-tuned model nor Ollama available. "
                "Install Ollama with: pip install ollama, or train fine-tuned model."
            )

        self.model = model

        # Ollama client setup
        if ollama_host is None:
            ollama_host = os.getenv('OLLAMA_HOST', 'localhost').split(',')[0]
        self.ollama_client = ollama.Client(host=f'http://{ollama_host}:11434')

        # Lazy-initialized embeddings
        self.safe_embeddings: Optional[np.ndarray] = None
        self.unsafe_embeddings: Optional[np.ndarray] = None

        # Embedding cache: {xml_string → embedding_vector}
        self._cache: dict[str, EmbeddingVector] = {}

        logger.info(
            f"Initialized EmbeddingClassifier (Ollama): model={model}, "
            f"thresholds=[{threshold_deny:.3f}, {threshold_allow:.3f}]"
        )

    def initialize(self) -> None:
        """
        Pre-compute anchor embeddings (blocking operation).

        Flow:
            SAFE_CONCEPTS → get_embedding() → safe_embeddings
            UNSAFE_CONCEPTS → get_embedding() → unsafe_embeddings

        Side Effects:
            - Sets self.safe_embeddings
            - Sets self.unsafe_embeddings
            - Populates self._cache with anchor embeddings

        Performance:
            - ~1-2 seconds for 47 anchors (18 safe + 29 unsafe)
            - Blocks until all embeddings computed
        """
        # Delegate to fine-tuned implementation if available
        if self._use_finetuned and self._impl is not None:
            self._impl.initialize()
            return
        
        if self.safe_embeddings is not None and self.unsafe_embeddings is not None:
            logger.debug("Embeddings already initialized, skipping")
            return

        logger.info(
            f"Initializing classifier with {len(self.SAFE_CONCEPTS)} safe "
            f"and {len(self.UNSAFE_CONCEPTS)} unsafe concepts..."
        )

        # Compute safe embeddings
        safe_embs: list[EmbeddingVector] = []
        for i, concept in enumerate(self.SAFE_CONCEPTS):
            if (i + 1) % 10 == 0:
                logger.debug(f"  Safe concepts: {i + 1}/{len(self.SAFE_CONCEPTS)}")
            emb = self._get_embedding(concept)
            if emb is not None:
                safe_embs.append(emb)

        # Compute unsafe embeddings
        unsafe_embs: list[EmbeddingVector] = []
        for i, concept in enumerate(self.UNSAFE_CONCEPTS):
            if (i + 1) % 10 == 0:
                logger.debug(f"  Unsafe concepts: {i + 1}/{len(self.UNSAFE_CONCEPTS)}")
            emb = self._get_embedding(concept)
            if emb is not None:
                unsafe_embs.append(emb)

        self.safe_embeddings = np.array(safe_embs)
        self.unsafe_embeddings = np.array(unsafe_embs)

        logger.info(
            f"✓ Classifier ready: {len(safe_embs)} safe, {len(unsafe_embs)} unsafe concepts"
        )

    def classify(self, xml_string: str) -> tuple[Decision, SafetyScore]:
        """
        Classify raw XML tool call with 1D safety score.

        Flow:
            xml_string
                ↓
            get_embedding()
                ↓
            avg_similarity(cmd_emb, safe_embeddings) → safe_sim
            avg_similarity(cmd_emb, unsafe_embeddings) → unsafe_sim
                ↓
            safety_score = (safe_sim - unsafe_sim) / (safe_sim + unsafe_sim)
                ↓
            threshold_comparison()
                ↓
            (decision, score)

        Args:
            xml_string: Raw XML tool call (e.g., "<bash>ls</bash>")

        Returns:
            Tuple of (decision, safety_score):
                - decision: "auto-allow" | "ask" | "neutral"
                - safety_score: -1.0 (unsafe) to +1.0 (safe)

        Decision Logic:
            score > threshold_allow  → "auto-allow"
            score < threshold_deny   → "ask"
            else                     → "neutral"

        Examples:
            >>> classifier.classify("<bash>ls</bash>")
            ("auto-allow", 0.042)

            >>> classifier.classify("<bash>rm -rf /</bash>")
            ("ask", -0.078)

            >>> classifier.classify("<readfile><filepath>./src/main.py</filepath></readfile>")
            ("neutral", -0.003)
        """
        # Delegate to fine-tuned implementation if available
        if self._use_finetuned and self._impl is not None:
            return self._impl.classify(xml_string)
        
        # Ensure initialized
        if self.safe_embeddings is None or self.unsafe_embeddings is None:
            self.initialize()

        # Get embedding for tool call
        cmd_embedding = self._get_embedding(xml_string)
        if cmd_embedding is None:
            logger.warning(f"Failed to get embedding for: {xml_string[:100]}")
            return "neutral", 0.0

        # Compute average similarity to safe group
        safe_sims = [
            self._cosine_similarity(cmd_embedding, safe_emb)
            for safe_emb in self.safe_embeddings
        ]
        avg_safe_sim = np.mean(safe_sims)

        # Compute average similarity to unsafe group
        unsafe_sims = [
            self._cosine_similarity(cmd_embedding, unsafe_emb)
            for unsafe_emb in self.unsafe_embeddings
        ]
        avg_unsafe_sim = np.mean(unsafe_sims)

        # Normalized difference: -1 to +1
        safety_score = (avg_safe_sim - avg_unsafe_sim) / (avg_safe_sim + avg_unsafe_sim + 1e-10)

        # Apply thresholds for decision
        if safety_score > self.threshold_allow:
            decision = "auto-allow"
        elif safety_score < self.threshold_deny:
            decision = "ask"
        else:
            decision = "neutral"

        logger.debug(
            f"Classification: {xml_string[:60]}... → {decision} "
            f"(score={safety_score:+.3f}, safe_sim={avg_safe_sim:.3f}, "
            f"unsafe_sim={avg_unsafe_sim:.3f})"
        )

        return decision, safety_score

    def _get_embedding(self, text: str) -> Optional[EmbeddingVector]:
        """
        Get embedding vector with caching.

        Flow:
            text → cache_lookup
                ├─ HIT → cached_embedding
                └─ MISS → ollama_api → embedding → cache_store → embedding

        Args:
            text: Text to embed (raw XML string)

        Returns:
            Embedding vector or None on failure

        Side Effects:
            - Updates self._cache on cache miss
        """
        # Check cache
        if text in self._cache:
            return self._cache[text]

        # Request embedding from Ollama
        try:
            response = self.ollama_client.embeddings(model=self.model, prompt=text)
            embedding = np.array(response['embedding'])

            # Store in cache
            self._cache[text] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding for '{text[:60]}...': {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: EmbeddingVector, b: EmbeddingVector) -> float:
        """
        Compute cosine similarity between two vectors.

        Formula:
            cosine_sim(a, b) = (a · b) / (||a|| * ||b||)

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Similarity score in range [-1, 1]
                1.0  = identical direction
                0.0  = orthogonal
                -1.0 = opposite direction
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return float(dot_product / (norm_a * norm_b + 1e-10))

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get embedding cache statistics.

        Returns:
            Dictionary with keys:
                - cache_size: Number of cached embeddings
                - anchor_count: Number of anchor embeddings (safe + unsafe)
        """
        # Delegate to fine-tuned implementation if available
        if self._use_finetuned and self._impl is not None:
            return self._impl.get_cache_stats()
        
        return {
            "cache_size": len(self._cache),
            "anchor_count": len(self.SAFE_CONCEPTS) + len(self.UNSAFE_CONCEPTS)
        }


def tool_call_to_xml(
    tool_name: str,
    file_path: Optional[str] = None,
    command: Optional[str] = None,
    xml_content: Optional[str] = None
) -> str:
    """
    Convert tool call parameters to raw XML format.

    Flow:
        (tool_name, file_path, command, xml_content)
            ↓
        xml_content provided? → return xml_content
            ↓ NO
        tool_name == "bash"? → <bash>{command}</bash>
            ↓ NO
        file_path provided? → <toolname><filepath>{file_path}</filepath></toolname>
            ↓ NO
        <toolname></toolname>

    Args:
        tool_name: Name of the tool (e.g., 'bash', 'readfile', 'writefile')
        file_path: Extracted file path if applicable
        command: Extracted bash command if tool is 'bash'
        xml_content: Raw XML if available (preferred)

    Returns:
        Raw XML string in LLM-generated format

    Examples:
        >>> tool_call_to_xml('bash', command='ls -la')
        '<bash>ls -la</bash>'

        >>> tool_call_to_xml('readfile', file_path='./src/main.py')
        '<readfile><filepath>./src/main.py</filepath></readfile>'

        >>> tool_call_to_xml('writefile', file_path='./config.json')
        '<writefile><filepath>./config.json</filepath></writefile>'

        >>> tool_call_to_xml('visualresponse')
        '<visualresponse></visualresponse>'
    """
    # Use raw XML if provided
    if xml_content:
        return xml_content

    # Bash tool: command goes directly in tag
    if tool_name == "bash" and command:
        return f"<bash>{command}</bash>"

    # File tools: use filepath tag
    if file_path:
        file_tools = {
            "readfile", "viewfiles", "editfile", "writefile",
            "appendfile", "insertline", "deletelines"
        }

        if tool_name in file_tools:
            return f"<{tool_name}><filepath>{file_path}</filepath></{tool_name}>"

        # Directory/search tools: use path tag
        path_tools = {"readdirectory", "grepsearch", "searchfile"}
        if tool_name in path_tools:
            return f"<{tool_name}><path>{file_path}</path></{tool_name}>"

    # Simple tools with no parameters
    return f"<{tool_name}></{tool_name}>"
