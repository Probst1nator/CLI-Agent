# Profile Validation System

This document describes the profile validation system that prevents corrupted (empty) profiles from being created or loaded.

## What is a Corrupted Profile?

A profile is considered corrupted/empty when **all** of the following are true:
- `override` is `None`
- `router` is `None`
- `planner` is `None`
- `fallbacks` is empty (`[]`)

A profile is **valid** if it has at least one non-None/non-empty value (router, planner, override, or fallbacks).

## Validation Points

### 1. Profile Load Prevention (`core/llm_profiles.py`)

**At Startup:**
```python
# Detects corrupted profiles
corrupted_profiles = manager.detect_corrupted_profiles()
# Warns user
# Prevents loading if active profile is corrupted
```

**Manual Load (`/profile_load`):**
```python
profile.apply()  # Raises ValueError if profile.is_empty()
```

**Result:** User sees error message and profile is not loaded.

### 2. Profile Save Prevention (`agent/commands/profile_commands.py`)

**Manual Save (`/profile_save`):**
```python
temp_profile = LLMProfile.from_current_config(profile_id)
if temp_profile.is_empty():
    return error: "Cannot save empty profile..."
```

**Result:** User must select at least one model before saving.

### 3. Auto-Save Prevention (`api/profile_api.py`)

**When LLM roles change via web UI:**
```python
temp_profile = LLMProfile.from_current_config(active_profile_id)
if temp_profile.is_empty():
    logger.warning("Skipping auto-save - would create empty profile")
    # Don't save
else:
    manager.update_profile(active_profile_id, from_current=True)
```

**Result:** Auto-save is skipped with warning in logs.

## User Experience

### Startup with Corrupted Profiles

```
⚠️  Found 1 corrupted profile(s) with no models: 1
    Use /profile_delete to remove them or /profile_save to overwrite them
⚠️  Active profile '1' is corrupted (all values are None/empty)
    Clearing active profile marker. Use /profile_load to load a valid profile.
```

### Attempting to Load Corrupted Profile

```
# cli-agent: Failed to load profile: Profile '1' is empty/corrupted -
all configuration values are None/empty
```

### Attempting to Save Empty Configuration

```
# cli-agent: Cannot save empty profile - all configuration values are None/empty.
Please select at least one model (router, planner, or fallback) before saving.
```

### In Web UI

- Corrupted profile slots show with red dashed border
- Pulsing red indicator dot
- Tooltip: "Corrupted profile (all values are None/empty). Click to overwrite or delete."

### In CLI (`/profile_list`)

```
  1 [CORRUPTED]
    ⚠️  All configuration values are None/empty
    Use /profile_delete to remove or /profile_save to overwrite
```

## Fixing Corrupted Profiles

### Option 1: Delete

```bash
/profile_delete 1
```

### Option 2: Overwrite

```bash
# 1. Select some models in the UI
# 2. Save to the corrupted profile slot
/profile_save 1
```

### Option 3: Ignore

Corrupted profiles are harmless - they just can't be loaded. You can leave them if you don't use those slots.

## Technical Implementation

### Detection Method (`core/llm_profiles.py:52-59`)

```python
def is_empty(self) -> bool:
    """Check if this profile is empty/corrupted (all None/empty values)."""
    return (
        self.override is None and
        self.router is None and
        self.planner is None and
        (not self.fallbacks or len(self.fallbacks) == 0)
    )
```

### Validation in Apply (`core/llm_profiles.py:61-73`)

```python
def apply(self) -> None:
    """Apply this profile to the current LLM configuration."""
    from core.globals import g

    if self.is_empty():
        raise ValueError(f"Cannot apply empty profile '{self.id}' - all configuration values are None/empty")

    # ... apply logic
```

### Auto-Save Validation (`api/profile_api.py:187-195`)

```python
# Check if the current configuration would create an empty/corrupted profile
temp_profile = LLMProfile.from_current_config(active_profile_id)
if temp_profile.is_empty():
    logger.warning(f"Skipping auto-save to profile '{active_profile_id}' - would create empty/corrupted profile")
else:
    manager.update_profile(active_profile_id, from_current=True)
    logger.info(f"Auto-saved changes to active profile '{active_profile_id}'")
```

## Edge Cases

### Partial Configuration

These are **valid** profiles:
- Only router set: ✅ Valid
- Only planner set: ✅ Valid
- Only fallbacks set: ✅ Valid
- Router + planner: ✅ Valid
- Any combination with at least one value: ✅ Valid

### Temporary Empty State

If a user deselects all models in the UI:
1. Runtime configuration becomes empty
2. Auto-save is **skipped** with warning
3. Profile retains last valid state
4. User can undo by selecting models again

### Profile 2 in Your Test

From your log:
```
🔧 LLM role 'fallbacks' updated to: None  # Temporarily empty
...
🔧 LLM role 'fallbacks' updated to: ['gemini-flash-latest']  # Restored
```

Profile 2 never became corrupted because:
1. It still had `router` and `planner` set
2. When fallbacks were cleared, `is_empty()` returned `False` (router/planner still set)
3. Auto-save proceeded normally

## Files Modified

- `core/llm_profiles.py` - Added `is_empty()`, validation, cleanup methods
- `main.py` - Startup detection and warning
- `agent/commands/profile_commands.py` - Manual save validation, list display
- `api/profile_api.py` - Auto-save validation, API response
- `frontend/js/main.js` - UI detection and styling
- `frontend/css/main.css` - Corrupted profile styling

## Related Issues

This fixes the issue where profiles could be saved with all None/empty values, which would then fail to load on startup, causing confusion and potentially breaking the user's workflow.
