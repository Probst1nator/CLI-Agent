"""
Docker Runner for --isolated Mode

Orchestrates Docker container execution for isolated CLI-Agent runs.
Provides transparent containerization with cached image builds.
"""

import os
import sys
import subprocess
import shutil
import hashlib
from pathlib import Path
from typing import List


class DockerRunner:
    """Manages Docker-based isolated execution of CLI-Agent"""

    IMAGE_NAME = "cli-agent:isolated-cache"
    CONTAINER_PREFIX = "cli-agent-isolated"

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.dockerfile = self.project_root / "docker_files" / "Dockerfile.isolated-exec"
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_dir = Path.home() / ".cli-agent"

    def check_docker_available(self) -> bool:
        """Check if Docker is installed and accessible"""
        return shutil.which("docker") is not None

    def print_status(self, message: str, emoji: str = "🐳"):
        """Print colored status message"""
        print(f"\033[0;34m{emoji} {message}\033[0m", file=sys.stderr)

    def print_error(self, message: str):
        """Print colored error message"""
        print(f"\033[0;31m❌ {message}\033[0m", file=sys.stderr)

    def get_requirements_hash(self) -> str:
        """Get hash of requirements.txt for cache invalidation"""
        if not self.requirements_file.exists():
            return "no-requirements"

        with open(self.requirements_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    def image_exists(self) -> bool:
        """Check if cached Docker image exists"""
        result = subprocess.run(
            ["docker", "images", "-q", self.IMAGE_NAME],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())

    def build_image(self, force: bool = False):
        """Build the cached Docker image"""
        if not force and self.image_exists():
            self.print_status("Using cached Docker image", "✅")
            return

        if not self.dockerfile.exists():
            self.print_error(f"Dockerfile not found: {self.dockerfile}")
            sys.exit(1)

        self.print_status("Building Docker image (first run only, ~2-3 min)...", "🔨")

        build_cmd = [
            "docker", "build",
            "-f", str(self.dockerfile),
            "-t", self.IMAGE_NAME,
            str(self.project_root)
        ]

        try:
            subprocess.run(build_cmd, check=True, cwd=str(self.project_root))
            self.print_status("Image built successfully! Future runs will start instantly.", "✅")
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to build Docker image: {e}")
            sys.exit(1)

    def sync_source_code(self, container_name: str):
        """Copy latest source code into container"""
        # Create a temporary container to copy files into
        copy_cmd = [
            "docker", "cp",
            str(self.project_root) + "/.",
            f"{container_name}:/cli-agent/"
        ]

        try:
            subprocess.run(copy_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to sync source code: {e}")
            raise

    def prepare_docker_run_command(self, args: List[str]) -> List[str]:
        """Prepare docker run command with all necessary flags"""

        # Filter out --isolated from args
        filtered_args = [arg for arg in args if arg != '--isolated']

        # Container name with unique PID
        container_name = f"{self.CONTAINER_PREFIX}-{os.getpid()}"

        # Base docker run command
        cmd = [
            "docker", "run",
            "-it",  # Interactive TTY
            "--rm",  # Auto-remove on exit
            "--name", container_name,
        ]

        # Mount config directory for persistence
        self.config_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend([
            "-v", f"{self.config_dir}:/home/cliagent/.cli-agent"
        ])

        # Mount current working directory as /workspace
        cwd = Path.cwd()
        cmd.extend([
            "-v", f"{cwd}:/workspace",
            "-w", "/workspace"
        ])

        # Forward environment file if it exists
        env_file = self.project_root / ".env"
        if env_file.exists():
            cmd.extend(["--env-file", str(env_file)])

        # Forward API key environment variables
        api_key_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "GROQ_API_KEY", "BRAVE_API_KEY", "NVIDIA_API_KEY"
        ]
        for var in api_key_vars:
            if var in os.environ:
                cmd.extend(["-e", f"{var}={os.environ[var]}"])

        # Check if GUI mode is requested
        if "--gui" in filtered_args or "--api" in filtered_args:
            # Forward port 7860 for web server
            cmd.extend(["-p", "7860:7860"])

        # Check if voice mode is requested (need audio devices)
        if "-v" in filtered_args or "--voice" in filtered_args or "-s" in filtered_args or "--speak" in filtered_args:
            # Try to mount audio devices (best effort)
            if Path("/dev/snd").exists():
                cmd.extend(["--device", "/dev/snd"])
            else:
                self.print_status("Warning: Audio devices not available in container", "⚠️")

        # Add image name
        cmd.append(self.IMAGE_NAME)

        # Add Python command
        cmd.extend(["python3", "main.py"])
        cmd.extend(filtered_args)

        return cmd, container_name

    def run_container(self, args: List[str]):
        """Run CLI-Agent in isolated Docker container"""

        # Ensure Docker is available
        if not self.check_docker_available():
            self.print_error("Docker is not installed or not in PATH")
            self.print_error("Install Docker: https://docs.docker.com/get-docker/")
            sys.exit(1)

        # Build image if needed
        self.build_image()

        # Prepare run command
        docker_cmd, container_name = self.prepare_docker_run_command(args)

        # Create a temporary container first to sync source code
        self.print_status("Starting CLI-Agent in isolated container...")

        # Create container without starting
        create_cmd = [
            "docker", "create",
            "--name", container_name,
            "-v", f"{self.config_dir}:/home/cliagent/.cli-agent",
            self.IMAGE_NAME,
            "sleep", "infinity"
        ]

        try:
            # Create container
            subprocess.run(create_cmd, check=True, capture_output=True)

            # Sync source code
            self.sync_source_code(container_name)

            # Remove the temporary container
            subprocess.run(["docker", "rm", container_name], capture_output=True)

        except subprocess.CalledProcessError:
            # Clean up on error
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        # Now run the actual container with synced code
        try:
            # Use os.execvp to replace current process with docker
            # This preserves signal handling and TTY behavior
            os.execvp("docker", docker_cmd)
        except Exception as e:
            self.print_error(f"Failed to start container: {e}")
            sys.exit(1)


def run_in_isolated_container(argv: List[str]):
    """
    Main entry point for --isolated mode

    Args:
        argv: sys.argv (full command line arguments)
    """
    # Determine project root (where main.py is located)
    main_py = Path(argv[0]).resolve()
    if main_py.name == 'main.py':
        project_root = main_py.parent
    else:
        # Assume current directory is project root
        project_root = Path.cwd()

    # Create runner
    runner = DockerRunner(str(project_root))

    # Extract arguments (skip script name)
    args = argv[1:]

    # Run in container (this will exec and never return)
    runner.run_container(args)


if __name__ == "__main__":
    run_in_isolated_container(sys.argv)
