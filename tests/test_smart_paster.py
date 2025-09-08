import sys
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Set up environment to import the modules
test_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(test_root))

from tools.file_copier.smart_paster import (
    parse_clipboard_for_paths_and_code,
    process_smart_request
)
from tools.file_copier.ai_path_finder import AIFixPath

# --- Setup for Test Project ---
TEST_PROJECT_DIR = Path(__file__).parent / "test_project"

class TestSmartPaster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a mock project structure for testing
        cls.project_dir = TEST_PROJECT_DIR
        cls.project_dir.mkdir(exist_ok=True)
        (cls.project_dir / "api").mkdir(exist_ok=True)
        (cls.project_dir / "src").mkdir(exist_ok=True)
        (cls.project_dir / "src" / "components").mkdir(exist_ok=True)

        cls.files = {
            "api/users.py": "def get_users(): pass",
            "api/models.py": "class User: pass",
            "src/utils.js": "function format(): return true;",
            "src/components/profile.jsx": "export default Profile;",
            "README.md": "# Test Project"
        }

        for path, content in cls.files.items():
            (cls.project_dir / path).write_text(content)

    @classmethod
    def tearDownClass(cls):
        # Clean up the mock project structure
        import shutil
        if cls.project_dir.exists():
            shutil.rmtree(cls.project_dir)

    # --- Tests for parse_clipboard_for_paths_and_code ---

    def test_direct_path_identification(self):
        message = "Please update api/users.py"
        found_paths, orphan_code = parse_clipboard_for_paths_and_code(message, str(self.project_dir))
        
        expected_abs_path = str(self.project_dir / "api" / "users.py")
        self.assertIn(expected_abs_path, found_paths)
        self.assertEqual(orphan_code, [])

    def test_no_paths_found(self):
        message = "This is a code block without a path:\n\nprint('hello world')"
        found_paths, orphan_code = parse_clipboard_for_paths_and_code(message, str(self.project_dir))
        
        self.assertEqual(found_paths, [])
        self.assertEqual(orphan_code, [message.strip()])

    def test_mixed_paths_and_code(self):
        message = "Update the profile.jsx with this: src/components/profile.jsx\n\nconst Profile = () => { /* code */ };"
        found_paths, orphan_code = parse_clipboard_for_paths_and_code(message, str(self.project_dir))

        expected_abs_path = str(self.project_dir / "src" / "components" / "profile.jsx")
        self.assertIn(expected_abs_path, found_paths)
        
        # We expect the parsing to separate the path, leaving two code blocks
        expected_orphan_code = [
            "Update the profile.jsx with this:",
            "const Profile = () => { /* code */ };"
        ]
        self.assertEqual(sorted(orphan_code), sorted(expected_orphan_code))

    def test_multiple_paths_and_duplicates(self):
        message = "I need api/users.py and api/models.py. Also, update api/users.py with new logic."
        found_paths, orphan_code = parse_clipboard_for_paths_and_code(message, str(self.project_dir))
        
        expected_paths = [
            str(self.project_dir / "api" / "users.py"),
            str(self.project_dir / "api" / "models.py")
        ]
        # The function should deduplicate and return unique absolute paths
        self.assertEqual(set(found_paths), set(expected_paths))
        self.assertEqual(orphan_code, [
            "I need",
            "and",
            "Also, update",
            "with new logic."
        ])

    def test_non_existent_path_treated_as_code(self):
        message = "This is a fake path: non_existent/file.txt\n\n```python\ncode = 123\n```"
        found_paths, orphan_code = parse_clipboard_for_paths_and_code(message, str(self.project_dir))
        
        self.assertEqual(found_paths, [])
        self.assertEqual(orphan_code, [message.strip()])

    # --- Tests for process_smart_request (Integration) ---

    @patch('tools.file_copier.smart_paster.handle_missing_filepaths', new_callable=AsyncMock)
    def test_smart_request_direct_and_ai_paths(self, mock_handle_missing_filepaths):
        # Mock the AI to find a path for the orphan code
        mock_handle_missing_filepaths.return_value = [("src/utils.js", "function format")]
        
        message = "Please gather api/users.py and add the definition of the format() function."
        
        # Run the async function
        import asyncio
        found_rel_paths = asyncio.run(process_smart_request(message, str(self.project_dir)))
        
        # Expected paths: api/users.py (direct) and src/utils.js (AI inferred)
        expected_paths = sorted(["api/users.py", "src/utils.js"])
        self.assertEqual(found_rel_paths, expected_paths)

# --- Tests for AIFixPath extraction logic ---

class TestAIFixPathExtraction(unittest.TestCase):
    
    def test_extract_filepath_from_json(self):
        response_text = '{"filepath": "api/users.py", "reason": "found in tree"}'
        path = AIFixPath._extract_filepath_from_response(response_text)
        self.assertEqual(path, "api/users.py")

    def test_extract_filepath_from_fenced_json(self):
        response_text = "```json\n{\"filepath\": \"src/components/profile.jsx\"}\n```"
        path = AIFixPath._extract_filepath_from_response(response_text)
        self.assertEqual(path, "src/components/profile.jsx")

    def test_extract_filepath_from_plaintext_regex(self):
        # Test case 1: Simple path
        response_text = "api/models.py"
        path = AIFixPath._extract_filepath_from_response(response_text)
        self.assertEqual(path, "api/models.py")

        # Test case 2: Path in quotes
        response_text = '"src/utils.js"'
        path = AIFixPath._extract_filepath_from_response(response_text)
        self.assertEqual(path, "src/utils.js")

    def test_invalid_path_candidates(self):
        # A simple string without a separator
        response_text = "users.py"
        self.assertIsNone(AIFixPath._extract_filepath_from_response(response_text))

        # A path that's clearly a markdown header or list item
        response_text = "## api/users.py"
        self.assertIsNone(AIFixPath._extract_filepath_from_response(response_text))

        # A sentence with a path in it (more than 4 words)
        response_text = "The path to the file is src/utils.js"
        self.assertIsNone(AIFixPath._extract_filepath_from_response(response_text))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
