# CLI-Agent Test Environment Documentation

## Overview

This document describes the Docker-based test environment for running CLI-Agent autonomously on any coding task. The environment can be used with any TEST_TASK.md file to evaluate the agent's capabilities on complex software development challenges.

## Test Objective

The test evaluates CLI-Agent's ability to:
1. Clone and analyze a large TypeScript codebase (VSCodium)
2. Set up a complex build environment (Node.js, dependencies)
3. Implement a multi-phase feature across backend and frontend
4. Integrate with external APIs (Open VSX Registry)
5. Build and verify the implementation

## Container Design Philosophy

### Minimal Base Environment
- **Base**: Ubuntu 22.04 LTS (stable, well-supported)
- **Core Tools**: git, curl, basic build tools
- **CLI-Agent**: Full installation with all dependencies
- **No Pre-installed Development Environment**: Node.js, VSCodium dependencies, and build tools are intentionally left for the agent to install

### Agent Autonomy
The agent must handle:
- Environment setup (Node.js installation, version selection)
- VSCodium repository cloning and build system understanding
- Dependency management and build configuration
- Code analysis and implementation planning
- Feature implementation across multiple files
- Build verification and testing

## Test Execution Flow

### Phase 1: Container Startup
```bash
docker run -v $(pwd)/holy_grail_test_files:/test_files cli-agent-vscodium-test
```

### Phase 2: Agent Execution
The container automatically runs:
```bash
python3 -l -a -m "inspect TEST_TASK.md and try to finish as many steps as you can to complete it."
```

### Phase 3: Result Collection
- Build logs and agent output captured
- Implementation artifacts preserved
- Success metrics calculated based on completed steps
- Exit code indicates overall test result

## Implementation Steps Agent Must Handle

Based on TEST_TASK.md, the agent should attempt:

1. **Environment Setup** (Agent Responsibility)
   - Clone VSCodium repository
   - Install Node.js and build dependencies
   - Set up development environment

2. **Code Analysis**
   - Locate extension management files
   - Identify data interfaces and UI components
   - Understand the existing marketplace integration

3. **Backend Implementation**
   - Modify extension data interfaces
   - Implement Open VSX API integration
   - Add rating/review data fetching logic

4. **Frontend Implementation**
   - Locate and modify UI templates
   - Add star rating components
   - Implement styling for rating display

5. **Verification**
   - Build VSCodium from source
   - Test functionality (if possible in headless environment)
   - Verify no build errors or regressions

## Success Criteria

The test measures:
- **Setup Success**: Can the agent properly configure the build environment?
- **Code Understanding**: Does the agent correctly identify target files?
- **Implementation Quality**: Are the code changes appropriate and functional?
- **Build Success**: Does the modified code compile without errors?
- **Feature Completeness**: How many of the required steps were completed?

## CI/CD Integration

### Test Automation
- Designed for headless execution in CI/CD pipelines
- No interactive elements or GUI dependencies
- Structured logging for automated analysis
- Deterministic exit codes for pass/fail evaluation

### Performance Metrics
- Total execution time
- Steps completed vs. total steps
- Build success rate
- Code quality indicators

### Artifact Collection
- Modified source files
- Build logs and outputs
- Agent conversation history
- Error logs and debugging information

## Container Configuration

### Resource Requirements
- **Memory**: 4GB minimum (for VSCodium build)
- **CPU**: 2+ cores recommended
- **Storage**: 10GB for repositories and build artifacts
- **Network**: Internet access for repository cloning and API calls

### Volume Mounts
- `/test_files`: Contains TEST_TASK.md and related files
- `/output`: Collects test results and artifacts
- `/workspace`: Agent working directory

## Expected Outcomes

This test serves as a comprehensive benchmark for CLI-Agent's software development capabilities:

### Best Case Scenario
- Complete environment setup
- Full feature implementation
- Successful build with working ratings display
- Clean integration with existing codebase

### Realistic Expectations
- Successful repository setup and basic understanding
- Partial implementation of rating data structures
- Some UI components implemented
- Build succeeds with minimal functionality

### Learning Opportunities
- Understanding of complex TypeScript codebases
- Multi-step software development workflows
- API integration patterns
- Build system management

## Future Enhancements

- Support for multiple VSCodium versions
- Parallel testing with different agent configurations
- Integration with real Open VSX API for testing
- UI testing capabilities (when headless testing improves)
- Performance optimization measurement

## Usage Instructions

### Basic Usage
1. **Place your TEST_TASK.md** in any directory (e.g., `./my_test_files/`)
2. **Build the container:**
   ```bash
   docker build -f Dockerfile.test -t cli-agent-test .
   ```
3. **Run the test:**
   ```bash
   docker run -v ./my_test_files:/test_files:ro -v ./test_output:/output cli-agent-test
   ```

### Using Docker Compose
1. **Update docker-compose.yml** to point to your test directory:
   ```yaml
   volumes:
     - ./your_test_directory:/test_files:ro
   ```
2. **Run:**
   ```bash
   docker-compose -f docker-compose.test.yml up
   ```

### Output
- All results are saved to `./test_output/`
- `agent_output.log` contains the complete agent conversation
- `test_summary.json` contains execution metadata
- Any files created by the agent are copied to output

## Usage in Development

This test environment also serves as:
- Development sandbox for any complex coding task
- Training environment for agent capability improvement
- Benchmark for measuring agent development progress
- Reference implementation for testing agent autonomy