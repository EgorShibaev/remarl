"""Trusted isolation regression, executed only inside the sandbox."""
import ctypes
import errno
import json
import os
import socket
import subprocess
import sys

def denied(fn):
    try:
        fn()
    except PermissionError:
        return
    raise AssertionError('Network operation was not rejected with EPERM')

for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        denied(lambda: socket.socket(family, kind))
denied(socket.socketpair)
libc = ctypes.CDLL(None, use_errno=True)
for syscall, args in [(41, (2, 1, 0)), (42, (-1, 0, 0)), (44, (-1, 0, 0, 0, 0, 0)),
                       (425, (0, 0)), (426, (-1, 0, 0, 0, 0, 0)), (427, (-1, 0, 0, 0)),
                       (308, (-1, 0)), (272, (0,)), (438, (0, 0, 0))]:
    ctypes.set_errno(0)
    assert libc.syscall(syscall, *args) == -1 and ctypes.get_errno() == errno.EPERM, syscall
# exec and child creation cannot remove the inherited kernel filter.
child = subprocess.run([sys.executable, '-I', '-c',
    'import socket\ntry: socket.socket()\nexcept PermissionError: pass\nelse: raise AssertionError("exec bypass")'],
    capture_output=True, timeout=3)
assert child.returncode == 0, child.stderr
status = open('/proc/self/status').read()
assert 'NoNewPrivs:\t1' in status and 'Seccomp:\t2' in status
assert 'CapEff:\t0000000000000000' in status
for path in ('/home', '/data', '/root', '/etc/ssh', '/sys'):
    assert not os.path.exists(path), path
for name in ('HF_TOKEN', 'WANDB_API_KEY', 'SSH_AUTH_SOCK'):
    assert name not in os.environ
try:
    open('/usr/forbidden-sandbox-write', 'w')
except OSError:
    pass
else:
    raise AssertionError('Host runtime is writable')
assert len([p for p in os.listdir('/proc') if p.isdigit()]) <= 4
for name in os.listdir('/proc/self/fd'):
    try:
        assert not os.readlink('/proc/self/fd/'+name).startswith('socket:')
    except FileNotFoundError:
        pass
print(json.dumps(dict(passed=True, socket_families_denied=True, raw_syscalls_denied=True,
    io_uring_denied=True, exec_inheritance=True, no_inherited_sockets=True,
    no_new_privs=True, capabilities_dropped=True, filesystem_and_pid_isolated=True)))
