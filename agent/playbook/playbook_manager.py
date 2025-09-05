import numpy as np
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from termcolor import colored

# This is a forward declaration to avoid a circular import.
# The actual ToolVectorDB object will be passed during initialization.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from infrastructure.vector_db.vector_database import ToolVectorDB

# --- Default Playbooks Database ---
# A more comprehensive set of default strategies for the agent.
# This list is used to bootstrap the playbook database if one doesn't exist.
DEFAULT_PLAYBOOKS = [
    {
        "name": "Website Scraping and Information Extraction",
        "trigger_intent": "scrape, download, or extract specific information (like files, data, text) from a website or URL",
        "thoughts": [
            "First, I must determine if the website offers a public API, which is always preferable to scraping.",
            "If no API is available, I will use `curl` or Python's `requests` library to fetch the website's raw HTML content.",
            "After fetching the HTML, I will use Python's `BeautifulSoup` or `lxml` to parse the document and locate the target elements (e.g., links, text, tables).",
            "I need to be mindful of the website's `robots.txt` file and terms of service to ensure my actions are compliant and respectful.",
            "Finally, I will save the extracted information to a local file or process it as requested."
        ]
    },
    {
        "name": "Large Log File Analysis",
        "trigger_intent": "process, search, filter, or analyze a large text or log file for patterns, errors, or specific entries",
        "thoughts": [
            "To avoid memory issues, I should not read the entire large file at once. I will process it line-by-line.",
            "For simple pattern matching and filtering, standard bash tools like `grep`, `awk`, and `sed` are extremely efficient and should be my first choice.",
            "If the analysis requires more complex logic or state-tracking, a Python script that iterates over the file handle is the better approach.",
            "I should start by examining the first few lines of the file (`head`) to understand its structure before writing my full script."
        ]
    },
    {
        "name": "Software Installation and Configuration",
        "trigger_intent": "install a new software package, tool, or library and configure it",
        "thoughts": [
            "I need to identify the correct package manager for the system (e.g., `apt`, `yum`, `pip`, `npm`). I can check this by trying the `--version` command for each.",
            "Before installation, I will search for the exact package name to ensure I'm installing the correct software.",
            "After installation, I must verify it was successful by checking the version or running a basic command.",
            "If configuration is required, I will look for default configuration files in common locations like `/etc/`, `~/.config/`, or the user's home directory."
        ]
    },
    {
        "name": "Code Debugging and Refinement",
        "trigger_intent": "debug a script, fix an error in code, or refactor a piece of software",
        "thoughts": [
            "First, I need to fully understand the error. I will read the error message and the surrounding code carefully.",
            "I will try to reproduce the error consistently in the current environment.",
            "I will form a hypothesis about the cause of the error and devise a minimal change to test it.",
            "For refactoring, I will identify the code to be improved and create a new, cleaner version, ensuring I don't change its core functionality unless requested."
        ]
    },
    {
        "name": "Automated Content Curation and Notification",
        "trigger_intent": "fetch content from an online source (like RSS, API), process it, and then send a summary or notification",
        "thoughts": [
            "First, I'll identify the best method to retrieve the data (e.g., `curl` an API endpoint, use a Python library for RSS).",
            "Next, I'll process the retrieved data to extract the key information needed for the summary.",
            "Then, I'll use an appropriate tool or model to summarize or format the extracted data.",
            "Finally, I'll deliver the result using the requested method, such as sending an email, posting to a webhook, or saving to a file."
        ]
    }
]


class PlaybookManager:
    """Manages strategic playbooks for complex, multi-step tasks."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlaybookManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, vector_db: 'ToolVectorDB'):
        if self._initialized:
            return

        self.vector_db = vector_db
        self.playbooks: List[Dict[str, Any]] = []

        if self.vector_db is None or not self.vector_db.is_ready:
            logging.warning("PlaybookManager could not initialize; VectorDB not ready.")
            return

        self._init_persistence()

        if not self._load_db():
            self._initialize_from_defaults()
            self._save_db()

        self._initialized = True
        logging.info(colored(f"  - PlaybookManager initialized with {len(self.playbooks)} playbooks.", "blue"))

    def _init_persistence(self):
        """Initialize path for playbook database persistence."""
        try:
            from core.globals import g
            cache_dir = Path(g.CLIAGENT_PERSISTENT_STORAGE_PATH)
            cache_dir.mkdir(exist_ok=True)
            self.db_path = cache_dir / "playbook_db.json"
        except Exception as e:
            logging.warning(f"Could not initialize persistent storage for playbooks: {e}")
            self.db_path = None

    def _initialize_from_defaults(self):
        """Loads playbooks from the hardcoded defaults and creates their embeddings."""
        logging.info("No existing playbook database found. Initializing from defaults.")
        for playbook in DEFAULT_PLAYBOOKS:
            trigger_intent = playbook.get("trigger_intent")
            if trigger_intent:
                embedding = self.vector_db._get_embedding(f"strategic playbook for: {trigger_intent}")
                playbook_data = playbook.copy()
                playbook_data["embedding"] = embedding
                self.playbooks.append(playbook_data)

    def _save_db(self):
        """Saves the current state of playbooks to a JSON file."""
        if not self.db_path:
            return

        try:
            serializable_playbooks = []
            for playbook in self.playbooks:
                p_copy = playbook.copy()
                if isinstance(p_copy.get("embedding"), np.ndarray):
                    p_copy["embedding"] = p_copy["embedding"].tolist()
                serializable_playbooks.append(p_copy)

            with open(self.db_path, 'w') as f:
                json.dump({"playbooks": serializable_playbooks}, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save playbook database: {e}")

    def _load_db(self) -> bool:
        """Loads playbooks from the JSON file."""
        if not self.db_path or not self.db_path.exists():
            return False

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f).get("playbooks", [])
            
            for s_playbook in data:
                d_playbook = s_playbook.copy()
                d_playbook["embedding"] = np.array(d_playbook["embedding"], dtype=np.float32)
                self.playbooks.append(d_playbook)
            
            from termcolor import colored
            logging.info(colored(f"  - Loaded {len(self.playbooks)} playbooks from persistent storage.", "cyan"))
            return True
        except Exception as e:
            logging.error(f"Failed to load playbook database: {e}. Re-initializing.")
            return False

    def get_relevant_playbook(self, query: str, threshold: float = 0.75) -> Optional[Dict[str, Any]]:
        """Finds a relevant playbook for a given query if it meets the similarity threshold."""
        if not self.playbooks or not self.vector_db.is_ready:
            return None

        # Reuse the vector_db's embedding and similarity calculation methods
        query_embedding = self.vector_db._get_embedding(query)
        
        best_match = None
        highest_score = -1.0

        all_playbook_embeddings = np.array([p["embedding"] for p in self.playbooks])
        
        # Calculate all similarities in one go
        similarity_matrix = self.vector_db.cos_sim(query_embedding.reshape(1, -1), all_playbook_embeddings)
        scores = similarity_matrix[0]
        
        # Get the index of the best match
        best_index = np.argmax(scores)
        highest_score = scores[best_index]

        if highest_score >= threshold:
            best_match = self.playbooks[best_index]
            logging.info(colored(f" Playbook matched: '{best_match['name']}' (Score: {highest_score:.2f})", "magenta"))
            return best_match
        
        return None
