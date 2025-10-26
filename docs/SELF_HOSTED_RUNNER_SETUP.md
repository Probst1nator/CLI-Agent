# Self-Hosted GitHub Actions Runner Setup

This guide will help you set up a self-hosted GitHub Actions runner on your local deployment server, so deployments happen automatically when you push code from anywhere.

## Benefits of Self-Hosted Runner

- ✅ **No external container registry needed** - Images build and stay local
- ✅ **Faster deployments** - No pushing/pulling images over internet
- ✅ **More secure** - Images never leave your network
- ✅ **Deploy from anywhere** - Push from any machine, runner handles deployment
- ✅ **Automatic testing** - Tests run before deployment
- ✅ **Free** - No GitHub Actions minutes used

## Prerequisites

- Your deployment server (192.168.178.20 or wherever you're deploying)
- Docker installed and running
- Admin access to your GitHub repository
- Network access to GitHub.com from your server

## Setup Steps

### 1. Navigate to Runner Settings

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/CLI-Agent`
2. Click **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Select your OS (likely **Linux**) and architecture (likely **x64** or **ARM64**)

### 2. Install the Runner on Your Server

GitHub will show you commands to run. They'll look something like this:

```bash
# SSH into your deployment server
ssh user@192.168.178.20

# Create a directory for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download the runner (GitHub provides the exact URL)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract the installer
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
```

### 3. Configure the Runner

```bash
# Configure the runner (GitHub provides the exact token)
./config.sh --url https://github.com/YOUR_USERNAME/CLI-Agent --token YOUR_TOKEN_HERE

# When prompted:
# - Runner name: cli-agent-local (or whatever you prefer)
# - Runner group: Default
# - Labels: self-hosted,Linux,X64 (default is fine)
# - Work folder: _work (default is fine)
```

### 4. Install as a Service (Recommended)

Running as a service ensures the runner starts automatically on boot:

```bash
# Install the service
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

**Alternative: Run interactively (for testing)**
```bash
# Run in foreground (useful for debugging)
./run.sh
```

### 5. Verify Runner is Connected

1. Go back to GitHub: **Settings** → **Actions** → **Runners**
2. You should see your runner listed with a green "Idle" status

### 6. Update Docker Compose to Build Locally

Make sure your `docker-compose.infrastructure.yml` is set to build images (not pull them):

```yaml
  frontend:
    build:
      context: ..
      dockerfile: docker_files/Dockerfile.frontend
    # ... rest of config

  container-manager:
    image: cli-agent-manager:latest  # Built by workflow
    # ... rest of config
```

### 7. Disable Old Workflows (Optional)

You may want to disable the old workflows that use GitHub Container Registry:

1. Go to **Actions** tab in GitHub
2. Click on the workflow (e.g., "Deploy to Staging")
3. Click the **"..."** menu → **Disable workflow**

Or rename the files to prevent them from running:
```bash
mv .github/workflows/deploy-staging.yml .github/workflows/deploy-staging.yml.disabled
mv .github/workflows/deploy-production.yml .github/workflows/deploy-production.yml.disabled
```

## Testing the Setup

1. Make a small change to your code
2. Commit and push to `main` or `develop` branch:
   ```bash
   git add .
   git commit -m "Test self-hosted runner deployment"
   git push origin main
   ```
3. Go to the **Actions** tab in GitHub
4. You should see the workflow running on your self-hosted runner
5. Check your deployment server to see the updated containers

## Monitoring the Runner

### Check runner status
```bash
sudo systemctl status actions.runner.YOUR-USERNAME-CLI-Agent.cli-agent-local.service
```

### View runner logs
```bash
sudo journalctl -u actions.runner.YOUR-USERNAME-CLI-Agent.cli-agent-local.service -f
```

### Restart runner
```bash
sudo ./svc.sh restart
```

## Troubleshooting

### Runner shows offline
- Check if the service is running: `sudo ./svc.sh status`
- Check network connectivity to GitHub: `ping github.com`
- Check logs: `sudo journalctl -u actions.runner.* -f`

### Deployment fails with permission errors
- Ensure the runner user has Docker permissions:
  ```bash
  sudo usermod -aG docker $(whoami)
  # Log out and back in for changes to take effect
  ```

### Docker commands fail
- Make sure Docker socket is accessible:
  ```bash
  ls -la /var/run/docker.sock
  docker ps  # Should list containers
  ```

### Environment variables not available
- The runner needs access to `.env` file or environment variables
- Option 1: Create `.env` file in repository root (don't commit secrets!)
- Option 2: Set environment variables in runner configuration
- Option 3: Use GitHub Secrets (for sensitive values)

## Security Considerations

### 1. Runner Isolation
- The runner has full access to your server
- Only add the runner to repositories you trust
- Consider running the runner in a Docker container or VM for additional isolation

### 2. Secrets Management
- Don't commit API keys or secrets to the repository
- Use GitHub Secrets for sensitive values:
  1. Go to **Settings** → **Secrets and variables** → **Actions**
  2. Add secrets (e.g., `ANTHROPIC_API_KEY`)
  3. Access in workflow: `${{ secrets.ANTHROPIC_API_KEY }}`

### 3. Network Security
- The runner needs outbound access to GitHub.com
- Consider firewall rules to restrict runner's network access
- Runner should only accept connections from localhost

## Maintenance

### Updating the Runner
GitHub will notify you when updates are available:

```bash
cd ~/actions-runner
sudo ./svc.sh stop
./config.sh remove  # Remove old configuration
# Download and extract new version (from GitHub instructions)
./config.sh --url https://github.com/YOUR_USERNAME/CLI-Agent --token NEW_TOKEN
sudo ./svc.sh install
sudo ./svc.sh start
```

### Removing the Runner
```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_TOKEN
```

## Workflow Configuration

The new workflow (`.github/workflows/deploy-local-runner.yml`) is optimized for self-hosted runners:

- **Runs on**: `self-hosted` label targets your local runner
- **No registry**: Builds images directly on your server
- **Fast**: No image push/pull over internet
- **Simple**: Just git pull, build, restart

## Next Steps

Once set up, your deployment workflow will be:

1. **Develop anywhere** → Make changes on any machine
2. **Push to GitHub** → `git push origin main`
3. **Automatic deployment** → Runner builds and deploys locally
4. **Verification** → Check `http://192.168.178.20` to see changes

That's it! Your local runner handles everything automatically.
