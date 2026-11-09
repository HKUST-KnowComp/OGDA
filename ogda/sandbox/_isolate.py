"""Linux Landlock + seccomp setup, run before any agent-written code.

This is a process entry point, intentionally independent of the method package.
It uses kernel controls, not monkey patches. Any failed restriction exits before
compiling or executing the supplied program.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import resource
import runpy
import sys
from pathlib import Path


def isolate(read_roots: list[str], write_roots: list[str], memory_mb: int, timeout: int) -> None:
    restrict_filesystem(read_roots, write_roots)
    restrict_syscalls()
    restrict_resources(memory_mb, timeout)


def restrict_filesystem(read_roots: list[str], write_roots: list[str]) -> None:
    """Apply before starting threads so all descendants inherit Landlock."""
    if sys.platform != "linux":
        raise RuntimeError("OS sandbox requires Linux Landlock and seccomp")
    libc = ctypes.CDLL(None, use_errno=True)
    # The syscall numbers are shared by x86_64/aarch64/riscv64.
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError(f"Landlock ABI >=3 required, got {abi}")
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    # EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE_DIR, REMOVE_FILE,
    # MAKE_CHAR/DIR/REG/SOCK/FIFO/BLOCK/SYM, REFER, TRUNCATE.
    all_access = (1 << 15) - 1
    read_access = (1 << 0) | (1 << 2) | (1 << 3)
    fd = libc.syscall(444, ctypes.byref(Ruleset(all_access)), ctypes.sizeof(Ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, permissions in [(p, read_access) for p in read_roots] + [(p, all_access) for p in write_roots]:
            target = Path(path)
            if not target.exists():
                continue
            allowed = permissions
            if not target.is_dir():
                allowed &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            parent = os.open(target, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(allowed, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)


def restrict_syscalls() -> None:
    """Deny networking and process escape before interpreter threads start."""
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO(EPERM)
        names = ["socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg", "ptrace", "process_vm_readv", "process_vm_writev", "open_by_handle_at", "mount", "umount2", "pivot_root", "setns", "unshare", "setsid", "setpgid", "bpf", "userfaultfd", "io_uring_setup", "kexec_load", "kill", "tkill"]
        for name in names:
            syscall = seccomp.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and seccomp.seccomp_rule_add(context, deny, syscall, 0) != 0:
                raise RuntimeError(f"seccomp rule failed: {name}")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def restrict_resources(memory_mb: int, timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    # NumPy/Arrow reserve substantial virtual address space even with 1 thread.
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2, memory_mb * 1024**2))


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    try:
        isolate(config["read_roots"], config["write_roots"], config["memory_mb"], config["timeout"])
    except BaseException as error:
        print(f"SANDBOX_SETUP_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(125)
    # The program's directory is useful for local helper imports, but remains
    # confined to explicit input/output roots by Landlock.
    sys.argv = [config["code_path"]]
    sys.path.insert(0, str(Path(config["code_path"]).parent))
    reads = set()
    def record_open(event, arguments):
        if event == "open" and arguments and isinstance(arguments[0], (str, bytes)):
            mode = arguments[1]
            if mode is None or "r" in str(mode) or "+" in str(mode):
                reads.add(os.path.abspath(os.fsdecode(arguments[0])))
    sys.addaudithook(record_open)
    try:
        runpy.run_path(config["code_path"], run_name="__main__")
    finally:
        print("SANDBOX_READ_LOG:" + json.dumps(sorted(reads)), file=sys.stderr)


if __name__ == "__main__":
    main()
