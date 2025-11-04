from typing import Optional

"""
Prompt Regenerator for CLI-Agent

Intelligently generates contextual prompt suggestions by analyzing
recently modified files in the user's Studium directory using LLM analysis.

This replaces the shell script approach with a Python implementation that
integrates properly with the CLI-Agent framework and logging system.
"""

import argparse
import asyncio
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptRegenerator:
    """
    Generates intelligent prompt suggestions based on recent study activities.
    """

    def __init__(
        self,
        studium_dir: Optional[str] = None,
        prompts_file: Optional[str] = None,
        num_files: int = 5,
        time_frame_days: int = 30,
    ):
        """
        Initialize the prompt regenerator.

        Args:
            studium_dir: Path to Studium directory (default: ~/OneDrive/Studium)
            prompts_file: Path to prompts JSON file (default: ~/.cli-agent/suggested_prompts.json)
            num_files: Number of random files to analyze (default: 5)
            time_frame_days: How many days back to look for modified files (default: 30)
        """
        self.studium_dir = Path(
            studium_dir or Path.home() / "OneDrive" / "Studium"
        ).expanduser()
        self.prompts_file = Path(
            prompts_file or Path.home() / ".cli-agent" / "suggested_prompts.json"
        ).expanduser()
        self.num_files = num_files
        self.time_frame_days = time_frame_days

        # Default prompts if generation fails
        self.default_prompts = [
            "Help me understand a complex concept from my recent studies",
            "Create practice problems based on my coursework",
            "Explain connections between different topics I'm learning",
            "Generate study questions for my upcoming exam",
            "Summarize key points from my recent lecture notes",
        ]

    def find_recent_files(self) -> list[tuple[Path, datetime]]:
        """
        Find recently modified files in Studium directory.

        Returns:
            List of (file_path, modification_time) tuples
        """
        if not self.studium_dir.exists():
            logger.warning(f"Studium directory not found: {self.studium_dir}")
            return []

        logger.info(
            f"Searching for files modified in last {self.time_frame_days} days in {self.studium_dir}"
        )

        # Use find command for better performance with large directories
        try:
            result = subprocess.run(
                [
                    "find",
                    str(self.studium_dir),
                    "-type",
                    "f",
                    "-mtime",
                    f"-{self.time_frame_days}",
                    "!",
                    "-path",
                    "*/.*",  # Exclude hidden files
                    "!",
                    "-name",
                    "*.exe",
                    "!",
                    "-name",
                    "*.dll",
                    "!",
                    "-name",
                    "*.so",
                    "!",
                    "-name",
                    "*.dylib",
                    "!",
                    "-name",
                    "*.zip",
                    "!",
                    "-name",
                    "*.tar.gz",
                    "!",
                    "-name",
                    "*.rar",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(f"find command failed: {result.stderr}")
                return []

            file_paths = [
                Path(line.strip())
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]

            # Get modification times
            files_with_mtime = []
            for file_path in file_paths:
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    files_with_mtime.append((file_path, mtime))
                except Exception as e:
                    logger.debug(f"Failed to get mtime for {file_path}: {e}")
                    continue

            logger.info(f"Found {len(files_with_mtime)} recent files")
            return files_with_mtime

        except FileNotFoundError:
            logger.error("'find' command not found. Please install findutils.")
            return []
        except Exception as e:
            logger.error(f"Error finding files: {e}")
            return []

    def select_random_files(
        self, files: list[tuple[Path, datetime]]
    ) -> list[tuple[Path, datetime]]:
        """
        Randomly select N files from the list.

        Args:
            files: List of (file_path, modification_time) tuples

        Returns:
            List of randomly selected files
        """
        if len(files) <= self.num_files:
            return files

        selected = random.sample(files, self.num_files)
        logger.info(f"Randomly selected {len(selected)} files for analysis")
        return selected

    def build_llm_prompt(self, files: list[tuple[Path, datetime]]) -> str:
        """
        Build the LLM prompt for generating contextual suggestions.

        Args:
            files: List of (file_path, modification_time) tuples

        Returns:
            Formatted prompt string
        """
        file_list = "\n".join(
            [
                f"- {file_path} (modified: {mtime.strftime('%Y-%m-%d')})"
                for file_path, mtime in files
            ]
        )

        prompt = f"""I need you to generate 5 contextual prompt suggestions based on my recent study activities.

## Context: Recently modified files
Here are randomly selected files from the last {self.time_frame_days} days:
{file_list}

## Your Task

1. **Understand my current focus**
   - Analyze the provided files to identify key topics and subjects
   - Explore the directory structure and read additional files within {self.studium_dir} to build richer context
   - Use your judgment on how much exploration is useful vs. excessive

2. **Generate 5 prompts**
   - Each prompt should be exactly one sentence
   - You can reference a file by title in your prompt but if you do so also add the full filepath at the end of it. (Ensure your sentence finished fully before providing the filepath as any ending filepaths will be hidden from the user)

3. **Output Format**
   Respond with ONLY a JSON object in this exact format:
   ```json
   {{
     "prompts": [
       "Prompt 1 here",
       "Prompt 2 here",
       "Prompt 3 here",
       "Prompt 4 here",
       "Prompt 5 here"
     ]
   }}
   ```

## Guidelines
- Be intelligent about inferring my real-world context and upcoming needs
- Think across subject boundaries - connect concepts from different domains
- Aim for prompts that are engaging and intellectually stimulating
- DO NOT include any text outside the JSON object"""

        return prompt

    async def generate_prompts_with_llm(
        self, files: list[tuple[Path, datetime]]
    ) -> Optional[list[str]]:
        """
        Use LLM to generate contextual prompts based on file analysis.

        Args:
            files: List of (file_path, modification_time) tuples

        Returns:
            List of generated prompts, or None if generation fails
        """
        try:
            from core.llm_router import LlmRouter

            logger.info("Generating prompts using LLM...")

            llm_router = LlmRouter()
            prompt = self.build_llm_prompt(files)

            # Print header for streamed output
            from termcolor import colored

            print(colored("\nGenerating prompts:\n", "cyan"))

            # Generate completion with streaming to console
            response = await llm_router.generate_completion(
                chat=prompt,  # Pass prompt directly as string
                generation_stream_callback=lambda char: print(
                    colored(char, "green"), end="", flush=True
                ),
                force_preferred_model=False,  # Allow fallback models
            )

            print(colored("\n\n✅ Generation complete!\n", "cyan"))

            if not response:
                logger.error("LLM returned empty response")
                return None

            # Response is a string, not a dict
            content = response
            logger.debug(f"LLM response: {content[:200]}...")

            # Extract JSON from response
            # Look for JSON block between ```json and ```
            json_match = content
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_match = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_match = content[start:end].strip()

            # Try to find JSON object if no code blocks
            if "{" in json_match and "}" in json_match:
                start = json_match.find("{")
                end = json_match.rfind("}") + 1
                json_match = json_match[start:end]

            # Parse JSON
            try:
                data = json.loads(json_match)
                prompts = data.get("prompts", [])

                if not prompts or not isinstance(prompts, list):
                    logger.error("LLM response missing 'prompts' array")
                    return None

                if len(prompts) != 5:
                    logger.warning(f"Expected 5 prompts, got {len(prompts)}")

                logger.info(f"Successfully generated {len(prompts)} prompts")
                return prompts

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"JSON content: {json_match}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import LLM dependencies: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating prompts with LLM: {e}", exc_info=True)
            return None

    def write_prompts_file(self, prompts: list[str]) -> bool:
        """
        Write prompts to JSON file.

        Args:
            prompts: List of prompt strings

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.prompts_file.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON file
            with open(self.prompts_file, "w", encoding="utf-8") as f:
                json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Successfully wrote {len(prompts)} prompts to {self.prompts_file}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to write prompts file: {e}")
            return False

    async def regenerate(self) -> dict[str, any]:
        """
        Main entry point: regenerate prompts based on recent study activity.

        Returns:
            Dict with status and prompts:
            {
                "status": "success" | "partial" | "error",
                "prompts": [...],
                "message": "...",
                "files_analyzed": N
            }
        """
        logger.info("Starting prompt regeneration...")
        start_time = datetime.now()

        # 1. Find recent files
        files = self.find_recent_files()
        if not files:
            logger.warning("No recent files found, using default prompts")
            self.write_prompts_file(self.default_prompts)
            return {
                "status": "partial",
                "prompts": self.default_prompts,
                "message": f"No files found in {self.studium_dir} (last {self.time_frame_days} days). Using default prompts.",
                "files_analyzed": 0,
            }

        # 2. Select random subset
        selected_files = self.select_random_files(files)

        # 3. Generate prompts with LLM
        prompts = await self.generate_prompts_with_llm(selected_files)
        if not prompts:
            logger.warning("LLM generation failed, using default prompts")
            self.write_prompts_file(self.default_prompts)
            return {
                "status": "partial",
                "prompts": self.default_prompts,
                "message": "LLM generation failed. Using default prompts.",
                "files_analyzed": len(selected_files),
            }

        # 4. Write to file
        if not self.write_prompts_file(prompts):
            return {
                "status": "error",
                "prompts": [],
                "message": "Failed to write prompts file",
                "files_analyzed": len(selected_files),
            }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prompt regeneration completed in {elapsed:.1f}s")

        return {
            "status": "success",
            "prompts": prompts,
            "message": f"Successfully generated {len(prompts)} prompts from {len(selected_files)} files",
            "files_analyzed": len(selected_files),
            "elapsed_seconds": elapsed,
        }


async def regenerate_prompts(
    studium_dir: Optional[str] = None, num_files: int = 5, time_frame_days: int = 30
) -> dict[str, any]:
    """
    Convenience function for API and programmatic access.

    Args:
        studium_dir: Path to Studium directory
        num_files: Number of files to analyze
        time_frame_days: Days to look back

    Returns:
        Result dictionary from PromptRegenerator.regenerate()
    """
    regenerator = PromptRegenerator(
        studium_dir=studium_dir, num_files=num_files, time_frame_days=time_frame_days
    )
    return await regenerator.regenerate()


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Regenerate CLI-Agent prompt suggestions from recent study files"
    )
    parser.add_argument(
        "--studium-dir",
        type=str,
        help="Path to Studium directory (default: ~/OneDrive/Studium)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of random files to analyze (default: 5)",
    )
    parser.add_argument(
        "--time-frame-days",
        type=int,
        default=30,
        help="Days to look back for modified files (default: 30)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run regeneration
    result = asyncio.run(
        regenerate_prompts(
            studium_dir=args.studium_dir,
            num_files=args.num_files,
            time_frame_days=args.time_frame_days,
        )
    )

    # Print results
    if result["status"] == "success":
        print(f"\n✅ {result['message']}")
        print("\nGenerated prompts:")
        for i, prompt in enumerate(result["prompts"], 1):
            print(f"  {i}. {prompt}")
    elif result["status"] == "partial":
        print(f"\n⚠️  {result['message']}")
        print("\nDefault prompts:")
        for i, prompt in enumerate(result["prompts"], 1):
            print(f"  {i}. {prompt}")
    else:
        print(f"\n❌ {result['message']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
