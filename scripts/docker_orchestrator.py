#!/usr/bin/env python3
"""
CLI-Agent Docker Orchestrator
Manages multiple containerized CLI-Agent instances with full directory synchronization
"""

import argparse
import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path
from typing import List, Optional
import yaml

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_status(color: str, message: str, file=sys.stderr):
    """Print colored status message"""
    print(f"{color}{message}{Colors.NC}", file=file)

class DockerOrchestrator:
    def __init__(self, cli_agent_dir: str, open_browser: bool = True, standalone: bool = False):
        self.cli_agent_dir = Path(cli_agent_dir).resolve()
        self.docker_dir = self.cli_agent_dir / "docker_files"
        self.compose_file = self.docker_dir / "docker-compose.sync.yml"
        self.image_name = "cli-agent:sync"
        self.container_prefix = "cli-agent-sync"
        self.network_name = "cli-agent-network"
        self.open_browser = open_browser
        self.standalone = standalone

    def build_image(self) -> bool:
        """Build the CLI-Agent sync Docker image"""
        print_status(Colors.BLUE, "🔨 Building CLI-Agent sync image...")

        dockerfile_path = self.docker_dir / "Dockerfile.sync"
        if not dockerfile_path.exists():
            print_status(Colors.RED, f"Error: Dockerfile not found at {dockerfile_path}")
            return False

        try:
            cmd = [
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", self.image_name,
                str(self.cli_agent_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.cli_agent_dir))

            if result.returncode == 0:
                print_status(Colors.GREEN, f"✅ Successfully built {self.image_name}")
                return True
            else:
                print_status(Colors.RED, "❌ Failed to build image:")
                print(result.stderr)
                return False

        except Exception as e:
            print_status(Colors.RED, f"❌ Error building image: {e}")
            return False

    def check_image_exists(self) -> bool:
        """Check if the CLI-Agent sync image exists"""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def generate_compose_file(self, count: int, extra_args: List[str] = None) -> bool:
        """Generate dynamic docker-compose file for N containers"""
        if extra_args is None:
            extra_args = []

        # Base port for each container (each gets 3 consecutive ports)
        base_port = 7860

        # Create compose structure
        compose_data = {
            'version': '3.8',
            'services': {},
            'volumes': {
                'cli-agent-data': {'driver': 'local'},
                'cli-agent-logs': {'driver': 'local'},
                'cli-agent-temp': {'driver': 'local'}
            },
            'networks': {
                self.network_name: {'driver': 'bridge'}
            }
        }

        # Generate services for each container
        for i in range(1, count + 1):
            service_name = f"{self.container_prefix}-{i}"
            container_port_base = base_port + (i - 1) * 3

            # Port mappings for this container
            ports = [
                f"{container_port_base}:7860",
                f"{container_port_base + 1}:7861",
                f"{container_port_base + 2}:7862"
            ]

            # Environment variables
            environment = [
                "PYTHONPATH=/workspace",
                "PYTHONUNBUFFERED=1",
                "CLI_AGENT_GUI_MODE=true",
                "CLI_AGENT_HOST=0.0.0.0",
                "CLI_AGENT_DATA_DIR=/workspace/data",
                "CLI_AGENT_LOG_DIR=/workspace/logs",
                "CLI_AGENT_TEMP_DIR=/workspace/temp",
                "DEBIAN_FRONTEND=noninteractive"
            ]

            # Add API keys from environment if available
            for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
                value = os.environ.get(key)
                if value:
                    environment.append(f"{key}={value}")

            # Configure volumes based on mode
            if self.standalone:
                # Standalone mode: only persistent data volumes, no source code mount
                volumes = [
                    f"cli-agent-data-{i}:/cli-agent/data",
                    f"cli-agent-logs-{i}:/cli-agent/logs",
                    f"cli-agent-temp-{i}:/cli-agent/temp"
                ]
                working_dir = '/cli-agent'
            else:
                # Sync mode: mount source code + persistent data volumes
                volumes = [
                    # Full directory sync - mount entire CLI-Agent directory
                    f"{self.cli_agent_dir}:/workspace",
                    # Persistent data volumes
                    f"cli-agent-data-{i}:/workspace/data",
                    f"cli-agent-logs-{i}:/workspace/logs",
                    f"cli-agent-temp-{i}:/workspace/temp"
                ]
                working_dir = '/workspace'

            # Service definition
            service_def = {
                'image': self.image_name,
                'container_name': service_name,
                'restart': 'unless-stopped',
                'ports': ports,
                'environment': environment,
                'volumes': volumes,
                'networks': [self.network_name],
                'working_dir': working_dir,
                'user': '1000:1000',  # Match host user
                'deploy': {
                    'resources': {
                        'limits': {'memory': '2G', 'cpus': '1.0'},
                        'reservations': {'memory': '512M', 'cpus': '0.25'}
                    }
                },
                'healthcheck': {
                    'test': [
                        'CMD-SHELL',
                        'curl -f http://localhost:7860/health || curl -f http://localhost:7861/health || curl -f http://localhost:7862/health || exit 1'
                    ],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '60s'
                },
                'logging': {
                    'driver': 'json-file',
                    'options': {'max-size': '10m', 'max-file': '3'}
                }
            }

            # Add extra arguments as command override
            if extra_args:
                service_def['command'] = extra_args

            compose_data['services'][service_name] = service_def

            # Add volume for this container
            compose_data['volumes'][f'cli-agent-data-{i}'] = {'driver': 'local'}
            compose_data['volumes'][f'cli-agent-logs-{i}'] = {'driver': 'local'}
            compose_data['volumes'][f'cli-agent-temp-{i}'] = {'driver': 'local'}

        # Write compose file
        try:
            self.docker_dir.mkdir(exist_ok=True)
            with open(self.compose_file, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)

            print_status(Colors.GREEN, f"✅ Generated compose file: {self.compose_file}")
            return True

        except Exception as e:
            print_status(Colors.RED, f"❌ Failed to generate compose file: {e}")
            return False

    def start_containers(self, count: int, extra_args: List[str] = None) -> bool:
        """Start N CLI-Agent containers"""
        print_status(Colors.BLUE, f"🚀 Starting {count} CLI-Agent container(s)...")

        # Check if image exists, build if not
        if not self.check_image_exists():
            print_status(Colors.YELLOW, "⚠️  Image not found, building...")
            if not self.build_image():
                return False

        # Generate compose file
        if not self.generate_compose_file(count, extra_args):
            return False

        # Start containers using docker-compose
        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "up", "-d"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.docker_dir))

            if result.returncode == 0:
                print_status(Colors.GREEN, f"✅ Successfully started {count} container(s)")

                # Show container info
                self.show_status()

                # Show access URLs and open browsers
                print_status(Colors.CYAN, "🌐 Access URLs:")
                base_port = 7860
                urls = []
                for i in range(1, count + 1):
                    port = base_port + (i - 1) * 3
                    url = f"http://localhost:{port}"
                    urls.append(url)
                    print_status(Colors.CYAN, f"  Container {i}: {url}")

                # Open browsers if requested
                if self.open_browser:
                    self._open_browsers_when_ready(urls, count)

                return True
            else:
                print_status(Colors.RED, "❌ Failed to start containers:")
                print(result.stderr)
                return False

        except Exception as e:
            print_status(Colors.RED, f"❌ Error starting containers: {e}")
            return False

    def _open_browsers_when_ready(self, urls: List[str], count: int):
        """Open browsers for each container when services are ready"""
        def wait_and_open():
            print_status(Colors.BLUE, "🌍 Waiting for services to be ready before opening browsers...")

            # Wait a bit for containers to fully start
            time.sleep(5)

            opened_count = 0
            max_wait_time = 120  # 2 minutes max wait
            start_time = time.time()

            for i, url in enumerate(urls, 1):
                print_status(Colors.BLUE, f"⏳ Waiting for container {i} service...")

                # Wait for this specific service to be ready
                service_ready = False
                while time.time() - start_time < max_wait_time:
                    try:
                        import urllib.request
                        with urllib.request.urlopen(url, timeout=5) as response:
                            if response.status == 200:
                                service_ready = True
                                break
                    except Exception:
                        pass
                    time.sleep(2)

                if service_ready:
                    try:
                        print_status(Colors.GREEN, f"🌍 Opening browser for container {i}: {url}")

                        # Try different browser opening methods
                        if sys.platform.startswith('linux'):
                            # Linux - try various methods
                            for cmd in ['xdg-open', 'firefox', 'google-chrome', 'chromium']:
                                try:
                                    subprocess.run([cmd, url],
                                                 check=False,
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL,
                                                 timeout=5)
                                    opened_count += 1
                                    break
                                except (subprocess.TimeoutExpired, FileNotFoundError):
                                    continue
                        else:
                            # macOS/Windows - use webbrowser module
                            webbrowser.open(url)
                            opened_count += 1

                        # Small delay between browser opens
                        time.sleep(1)

                    except Exception as e:
                        print_status(Colors.YELLOW, f"⚠️  Could not open browser for container {i}: {e}")
                else:
                    print_status(Colors.YELLOW, f"⚠️  Container {i} service not ready after {max_wait_time}s")

            if opened_count > 0:
                print_status(Colors.GREEN, f"✅ Opened {opened_count}/{count} browser windows")
            else:
                print_status(Colors.YELLOW, "⚠️  Could not open any browsers automatically")
                print_status(Colors.CYAN, "💡 You can manually open the URLs shown above")

        # Run browser opening in background thread
        browser_thread = threading.Thread(target=wait_and_open, daemon=True)
        browser_thread.start()

    def stop_containers(self) -> bool:
        """Stop all CLI-Agent containers"""
        print_status(Colors.BLUE, "🛑 Stopping CLI-Agent containers...")

        if not self.compose_file.exists():
            print_status(Colors.YELLOW, "⚠️  No compose file found, stopping containers by name...")
            return self._stop_containers_by_name()

        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "down"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.docker_dir))

            if result.returncode == 0:
                print_status(Colors.GREEN, "✅ Successfully stopped containers")
                return True
            else:
                print_status(Colors.RED, "❌ Failed to stop containers:")
                print(result.stderr)
                return False

        except Exception as e:
            print_status(Colors.RED, f"❌ Error stopping containers: {e}")
            return False

    def _stop_containers_by_name(self) -> bool:
        """Stop containers by name pattern"""
        try:
            # Get running containers with our prefix
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_prefix}", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print_status(Colors.RED, "❌ Failed to list containers")
                return False

            container_names = result.stdout.strip().split('\n')
            container_names = [name for name in container_names if name]  # Remove empty lines

            if not container_names:
                print_status(Colors.YELLOW, "⚠️  No CLI-Agent containers found")
                return True

            # Stop each container
            for container_name in container_names:
                print_status(Colors.BLUE, f"Stopping {container_name}...")
                subprocess.run(["docker", "stop", container_name], capture_output=True)

            print_status(Colors.GREEN, f"✅ Stopped {len(container_names)} container(s)")
            return True

        except Exception as e:
            print_status(Colors.RED, f"❌ Error stopping containers by name: {e}")
            return False

    def show_status(self) -> bool:
        """Show status of CLI-Agent containers"""
        try:
            # Get containers with our prefix
            result = subprocess.run([
                "docker", "ps", "-a",
                "--filter", f"name={self.container_prefix}",
                "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.RunningFor}}"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print_status(Colors.RED, "❌ Failed to get container status")
                return False

            output = result.stdout.strip()
            if not output or len(output.split('\n')) == 1:  # Only header or empty
                print_status(Colors.YELLOW, "⚠️  No CLI-Agent containers found")
                return True

            print_status(Colors.BLUE, "📊 CLI-Agent Container Status:")
            print(output)
            return True

        except Exception as e:
            print_status(Colors.RED, f"❌ Error getting status: {e}")
            return False

    def show_logs(self, container_number: Optional[int] = None) -> bool:
        """Show logs for containers"""
        try:
            if container_number:
                container_name = f"{self.container_prefix}-{container_number}"
                print_status(Colors.BLUE, f"📝 Logs for {container_name}:")
                result = subprocess.run(
                    ["docker", "logs", "--tail", "50", container_name],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
                    if result.stderr:
                        print_status(Colors.YELLOW, "STDERR:")
                        print(result.stderr)
                else:
                    print_status(Colors.RED, f"❌ Failed to get logs for {container_name}")
                    return False
            else:
                # Show logs for all containers
                result = subprocess.run([
                    "docker", "ps", "--filter", f"name={self.container_prefix}",
                    "--format", "{{.Names}}"
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print_status(Colors.RED, "❌ Failed to list containers")
                    return False

                container_names = result.stdout.strip().split('\n')
                container_names = [name for name in container_names if name]

                if not container_names:
                    print_status(Colors.YELLOW, "⚠️  No CLI-Agent containers found")
                    return True

                for container_name in container_names:
                    print_status(Colors.BLUE, f"📝 Logs for {container_name}:")
                    subprocess.run(["docker", "logs", "--tail", "20", container_name])
                    print("-" * 50)

            return True

        except Exception as e:
            print_status(Colors.RED, f"❌ Error getting logs: {e}")
            return False

    def clean_resources(self) -> bool:
        """Clean up Docker resources"""
        print_status(Colors.BLUE, "🧹 Cleaning up CLI-Agent Docker resources...")

        try:
            # Stop containers first
            self.stop_containers()

            # Remove stopped containers
            result = subprocess.run([
                "docker", "container", "prune", "-f",
                "--filter", "label=com.docker.compose.project=cli-agent"
            ], capture_output=True, text=True)

            # Remove unused volumes
            subprocess.run([
                "docker", "volume", "prune", "-f"
            ], capture_output=True, text=True)

            # Remove compose file
            if self.compose_file.exists():
                self.compose_file.unlink()
                print_status(Colors.GREEN, "✅ Removed compose file")

            print_status(Colors.GREEN, "✅ Cleanup completed")
            return True

        except Exception as e:
            print_status(Colors.RED, f"❌ Error during cleanup: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="CLI-Agent Docker Orchestrator")
    parser.add_argument("--count", type=int, help="Number of containers to start")
    parser.add_argument("--status", action="store_true", help="Show container status")
    parser.add_argument("--stop", action="store_true", help="Stop all containers")
    parser.add_argument("--logs", nargs="?", const=0, type=int, help="Show logs (optionally for specific container)")
    parser.add_argument("--clean", action="store_true", help="Clean up resources")
    parser.add_argument("--build", action="store_true", help="Build the Docker image")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browsers")
    parser.add_argument("--standalone", action="store_true", help="Use standalone mode (no bind mount of source code)")

    # Parse known args to allow passing extra arguments to containers
    args, extra_args = parser.parse_known_args()

    # Determine CLI-Agent directory
    cli_agent_dir = os.environ.get("CLI_AGENT_DIR", "~/repos/CLI-Agent")
    if not os.path.exists(cli_agent_dir):
        print_status(Colors.RED, f"❌ CLI-Agent directory not found: {cli_agent_dir}")
        sys.exit(1)

    # Create orchestrator with browser and standalone settings
    open_browser = not args.no_browser
    standalone = args.standalone
    orchestrator = DockerOrchestrator(cli_agent_dir, open_browser=open_browser, standalone=standalone)

    try:
        if args.build:
            success = orchestrator.build_image()
        elif args.status:
            success = orchestrator.show_status()
        elif args.stop:
            success = orchestrator.stop_containers()
        elif args.logs is not None:
            container_num = args.logs if args.logs > 0 else None
            success = orchestrator.show_logs(container_num)
        elif args.clean:
            success = orchestrator.clean_resources()
        elif args.count:
            if args.count < 1:
                print_status(Colors.RED, "❌ Container count must be at least 1")
                sys.exit(1)
            success = orchestrator.start_containers(args.count, extra_args)
        else:
            parser.print_help()
            sys.exit(1)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print_status(Colors.YELLOW, "\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_status(Colors.RED, f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()