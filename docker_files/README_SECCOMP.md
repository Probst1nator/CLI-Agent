# Seccomp Security Profile

## Overview

CLI-Agent user containers use Docker's **default seccomp profile** which blocks dangerous syscalls that could be used for container escape or system compromise.

## What is Seccomp?

Seccomp (Secure Computing Mode) is a Linux kernel feature that restricts the system calls a process can make. By blocking dangerous syscalls, we prevent:
- Container escapes
- Kernel exploits
- Privilege escalation
- System manipulation

## Docker's Default Seccomp Profile

Docker's default seccomp profile blocks ~44 dangerous syscalls including:

### Blocked Syscalls (Examples)

**Kernel Module Operations:**
- `init_module`, `finit_module`, `delete_module` - Load/unload kernel modules
- `create_module`, `query_module`, `get_kernel_syms` - Module management

**Filesystem Operations:**
- `mount`, `umount`, `umount2` - Mount/unmount filesystems
- `pivot_root` - Change root filesystem
- `swapon`, `swapoff` - Manage swap

**System Control:**
- `reboot` - Reboot the system
- `sethostname`, `setdomainname` - Change hostname/domain
- `settimeofday`, `clock_settime` - Change system time

**Process Manipulation:**
- `ptrace` - Debug/trace other processes
- `process_vm_readv`, `process_vm_writev` - Read/write other process memory

**Namespace/Virtualization:**
- `setns`, `unshare` - Namespace manipulation
- `clone` (with certain flags) - Process/namespace cloning

**Performance/Monitoring:**
- `perf_event_open` - Performance monitoring
- `bpf` - BPF program manipulation

**Kernel Communication:**
- `kexec_load`, `kexec_file_load` - Load new kernel
- `syslog` - Read kernel logs
- `_sysctl` - Read/write kernel parameters

## Custom Seccomp Profile

A custom seccomp profile has been created at `docker_files/seccomp-profile.json` which explicitly documents all blocked syscalls. To use it:

### Option 1: Mount in Container Manager (Recommended for Production)

1. **Update `docker-compose.infrastructure.yml`:**
   ```yaml
   container-manager:
     volumes:
       - /var/run/docker.sock:/var/run/docker.sock
       - ./docker_files/seccomp-profile.json:/etc/cli-agent/seccomp-profile.json:ro
   ```

2. **Update `api/container_manager.py`:**
   ```python
   security_opt=[
       "no-new-privileges:true",
       "apparmor=docker-default",
       "seccomp=/etc/cli-agent/seccomp-profile.json"  # Custom profile
   ],
   ```

### Option 2: Use Docker's Default (Current Implementation)

The current implementation uses Docker's default seccomp profile, which is sufficient for most security needs. The default profile blocks all dangerous syscalls listed above.

```python
# Current implementation (uses Docker's default seccomp)
security_opt=[
    "no-new-privileges:true",
    "apparmor=docker-default"
    # No seccomp specified = uses Docker default
],
```

## Security Impact

**With Seccomp Enabled:**
- ✅ Blocks container escape attempts via syscalls
- ✅ Prevents kernel module loading
- ✅ Blocks filesystem mounting attacks
- ✅ Prevents process debugging/injection
- ✅ Blocks system time manipulation
- ✅ Prevents namespace manipulation

**Allowed Operations:**
- ✅ Normal file I/O
- ✅ Network operations
- ✅ Process creation (fork, exec)
- ✅ Memory management
- ✅ Signal handling
- ✅ Timer operations

## Testing Seccomp

Test that dangerous syscalls are blocked:

```bash
# Try to mount (should fail)
docker exec <container-id> mount -t tmpfs tmpfs /tmp/test
# Error: Operation not permitted

# Try to load kernel module (should fail)
docker exec <container-id> modprobe test
# Error: Operation not permitted

# Try to reboot (should fail)
docker exec <container-id> reboot
# Error: Operation not permitted
```

## Troubleshooting

### Application fails with "Operation not permitted"

Some applications may require syscalls that are blocked by the seccomp profile. Options:

1. **Verify it's actually needed** - Often the syscall isn't required
2. **Use Docker's default profile** - More permissive
3. **Create custom profile** - Add specific syscalls back
4. **Run without seccomp** - ⚠️ NOT RECOMMENDED for production

### Disable seccomp (debugging only)

```python
security_opt=[
    "no-new-privileges:true",
    "apparmor=docker-default",
    "seccomp=unconfined"  # ⚠️ INSECURE - debugging only
],
```

## Related Security Features

This is one layer of defense in depth:

1. **Seccomp** - Blocks dangerous syscalls
2. **AppArmor** - Mandatory access control
3. **Capabilities** - Removed dangerous Linux capabilities
4. **no-new-privileges** - Prevents privilege escalation
5. **Non-root user** - Container Manager runs as UID 1000
6. **Resource limits** - CPU, memory, PID, disk I/O
7. **Network isolation** - Per-user isolated networks

## References

- [Docker Seccomp Documentation](https://docs.docker.com/engine/security/seccomp/)
- [Docker Default Seccomp Profile](https://github.com/moby/moby/blob/master/profiles/seccomp/default.json)
- [Linux Seccomp Manual](https://man7.org/linux/man-pages/man2/seccomp.2.html)
- [Seccomp Security Profile Specification](https://github.com/opencontainers/runtime-spec/blob/main/config-linux.md#seccomp)
