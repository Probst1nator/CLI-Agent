# Pull Request: [Brief Description]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] **Public sync** (main → public-sync)
- [ ] Refactoring
- [ ] Performance improvement

## Description
<!-- Provide a clear description of the changes -->

## Changes Made
<!-- List the main changes -->
-
-
-

---

## For Public Sync PRs (main → public-sync)

**⚠️ This PR proposes changes for the public repository**

### Public Release Checklist

#### Security & Privacy
- [ ] **No secrets**: No API keys, passwords, tokens, or credentials
- [ ] **No private infrastructure**: No references to saas_scripts, internal deployment details
- [ ] **No private data**: No user data, logs, or sensitive information
- [ ] **Environment variables**: All secrets properly configured in `.env.example` (not `.env`)

#### Code Quality
- [ ] **Safe tools only**: Any new privileged tools (bash, write operations) are intentional
- [ ] **No hardcoded paths**: No absolute paths specific to private environment
- [ ] **Dependencies documented**: New dependencies added to `requirements.txt`
- [ ] **Tests pass**: `python run_tests.py --quick` succeeds

#### Documentation
- [ ] **README appropriate**: Public-facing README is suitable for external users
- [ ] **Comments clean**: No internal comments about private infrastructure
- [ ] **Examples work**: Code examples/demos work in public version
- [ ] **CLAUDE.md updated**: If architecture changed, CLAUDE.md reflects it

#### Value Addition
- [ ] **Benefits public users**: Changes add value to the public repository
- [ ] **No breaking changes**: Existing public users won't be affected
- [ ] **Backwards compatible**: Or migration path documented

### Sensitive Content Check

**I have searched for and removed:**
- [ ] API keys and tokens
- [ ] Database credentials
- [ ] Internal URLs and endpoints
- [ ] Email addresses (except public contact)
- [ ] Internal team names/references
- [ ] Deployment-specific configuration

**Search commands run:**
```bash
# Search for common secret patterns
grep -r "API_KEY\|SECRET\|PASSWORD\|TOKEN" . --exclude-dir=.git

# Search for private infrastructure
grep -r "saas_scripts\|traefik\|deployment_logs" . --exclude-dir=.git

# Verify .env not committed
git diff main --name-only | grep "\.env$"
```

---

## Testing

### How Has This Been Tested?
<!-- Describe the tests you ran -->

**Test Environment:**
- OS:
- Python Version:
- Key Dependencies:

**Test Commands:**
```bash
python run_tests.py --quick
# Add any specific test commands here
```

### For Public Sync: Verification Steps
- [ ] Ran `python user_scripts/sync_public_repo.py --dry-run` successfully
- [ ] Verified filtered version works: `git checkout public-release && python main.py`
- [ ] Checked diff: `git diff public-sync..public-release` shows expected filtering

---

## Additional Context
<!-- Add any other context, screenshots, or information -->

## Related Issues
<!-- Link related issues: Fixes #123, Related to #456 -->

---

## Reviewer Notes

### For Maintainers Reviewing Public Sync PRs

**Priority Review Items:**
1. **Secrets Scan**: Use `git diff main..public-sync | grep -i "api\|key\|secret\|password"`
2. **Private References**: Check for saas_scripts, internal deployment mentions
3. **Tool Changes**: Verify any new privileged tools are intentional
4. **Documentation**: Ensure public-facing docs are appropriate

**Approval Criteria:**
- All security checklist items completed
- No sensitive information exposed
- Changes benefit public repository
- Backwards compatible or documented migration

**After Approval:**
The next steps will be:
1. Merge this PR to `public-sync`
2. Run `python user_scripts/sync_public_repo.py --prepare` from `public-sync` branch
3. Review filtered `public-release` branch
4. Push to public repository with `python user_scripts/sync_public_repo.py --push`

See `docs/RELEASE_WORKFLOW.md` for complete workflow.
