"""Export a native-x86_64 kernel filter; never load it into the trainer process."""

import ctypes as C
import errno
import hashlib
import json
from pathlib import Path
import platform

DENIED = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "sendmmsg",
    "recvmmsg",
    "shutdown",
    "getsockname",
    "getpeername",
    "setsockopt",
    "getsockopt",
    # io_uring can submit network operations without the corresponding syscalls.
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    # Prevent acquiring external descriptors or rearranging sandbox isolation.
    "pidfd_getfd",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "bpf",
    "setns",
    "unshare",
    "mount",
    "umount2",
    "pivot_root",
    "open_by_handle_at",
)


def build(base):
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    lib = C.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [C.c_uint32]
    lib.seccomp_init.restype = C.c_void_p
    lib.seccomp_release.argtypes = [C.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = C.c_int
    lib.seccomp_rule_add_array.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint, C.c_void_p]
    lib.seccomp_rule_add_array.restype = C.c_int
    lib.seccomp_export_bpf.argtypes = [C.c_void_p, C.c_int]
    lib.seccomp_export_pfc.argtypes = [C.c_void_p, C.c_int]
    # Native architecture only: libseccomp kills other architectures, including
    # x32 syscall-number variants. The filter is inherited across fork/exec.
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    numbers = {}
    try:
        for name in DENIED:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            assert number >= 0, ("Required syscall unavailable", name)
            rc = lib.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM, number, 0, None)
            assert rc == 0, (name, rc)
            numbers[name] = number
        for name, export in [
            ("network-deny.bpf", lib.seccomp_export_bpf),
            ("network-deny.pfc", lib.seccomp_export_pfc),
        ]:
            with (base / name).open("xb") as stream:
                assert export(context, stream.fileno()) == 0
    finally:
        lib.seccomp_release(context)
    bpf = (base / "network-deny.bpf").read_bytes()
    pfc = (base / "network-deny.pfc").read_text()
    assert len(bpf) > 0 and len(bpf) % 8 == 0
    assert "invalid architecture" in pfc and "KILL" in pfc
    receipt = dict(
        architecture="x86_64",
        denied_syscalls=numbers,
        bpf_sha256=hashlib.sha256(bpf).hexdigest(),
        default_action="ALLOW",
        network_action="ERRNO(EPERM)",
        invalid_architecture="KILL",
        no_network_namespace=True,
        no_fallback=True,
    )
    (base / "network-deny.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("SECCOMP_FILTER_BUILT", receipt["bpf_sha256"], flush=True)
    return receipt


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
