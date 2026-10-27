"""Read-only filesystem/PID isolation plus kernel-enforced network syscall denial."""

import json
import hashlib
import os
import platform
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time

BASE = Path(__file__).resolve().parent
PYTHON_BASE = Path(sys.base_prefix)
STATE = Path(os.environ.get("REMARL_SANDBOX_DIR", Path.home() / ".cache/remarl/sandbox"))


def filter_path():
    path = STATE / "network-deny.bpf"
    receipt = json.loads((STATE / "network-deny.json").read_text())
    assert receipt["architecture"] == "x86_64" and receipt["no_fallback"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["bpf_sha256"]
    return path


def preserve_failure(program, details):
    """Retain sandbox infrastructure failures for diagnosis."""
    digest = hashlib.sha256(program.encode()).hexdigest()
    folder = STATE / "execution-failures" / digest
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "program.py").write_text(program)
    (folder / f"{time.time_ns()}.json").write_text(json.dumps(details, indent=2))


def command(work, seccomp_fd, entry="/entry.py"):
    assert isinstance(seccomp_fd, int) and seccomp_fd >= 3
    system_mounts = [
        item
        for path in ("/usr", "/bin", "/lib", "/lib64")
        if Path(path).exists()
        for item in ("--ro-bind", path, path)
    ]
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return [
        "bwrap",
        "--unshare-all",
        "--share-net",
        "--unshare-user",
        "--seccomp",
        str(seccomp_fd),
        "--cap-drop",
        "ALL",
        "--die-with-parent",
        "--new-session",
        *system_mounts,
        "--ro-bind",
        str(PYTHON_BASE),
        "/opt/python",
        "--ro-bind",
        sysconfig.get_paths()["purelib"],
        "/opt/extra",
        "--ro-bind",
        str(BASE / "official"),
        "/runner",
        "--ro-bind",
        str(BASE / "sandbox_runner.py"),
        "/entry.py",
        "--ro-bind",
        str(work),
        "/work",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--clearenv",
        "--setenv",
        "PATH",
        "/opt/python/bin:/usr/bin:/bin",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--chdir",
        "/work",
        f"/opt/python/bin/{version}",
        "-I",
        entry,
    ]


def run(program):
    # Scratch contains ONLY this program, never credentials, checkpoints or other outputs.
    with tempfile.TemporaryDirectory(prefix="code-", dir=STATE / "scratch") as folder:
        work = Path(folder)
        (work / "program.py").write_text(program)
        started = time.monotonic()
        with filter_path().open("rb") as filter_stream:
            process = subprocess.Popen(
                command(work, filter_stream.fileno()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(filter_stream.fileno(),),
                close_fds=True,
                env={"PATH": "/usr/bin:/bin"},
                start_new_session=True,
            )
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            # The PID namespace's init is killed: its descendants cannot survive.
            preserve_failure(
                program, dict(kind="supervisor_outer_timeout", seconds=time.monotonic() - started)
            )
            return dict(passed=False, result="outer timeout", seconds=time.monotonic() - started)
        if process.returncode:
            preserve_failure(
                program,
                dict(
                    kind="supervisor_exit",
                    returncode=process.returncode,
                    stderr=stderr[-4000:].decode(errors="replace"),
                    seconds=time.monotonic() - started,
                ),
            )
            raise RuntimeError(
                f"sandbox infrastructure exit={process.returncode}: {stderr[-1000:]!r}"
            )
        try:
            result = json.loads(stdout)
            assert type(result["passed"]) is bool and isinstance(result["result"], str)
            assert result["passed"] == (result["result"] == "passed")
            assert result["result"] in ("passed", "timed out") or result["result"].startswith(
                "failed:"
            )
        except Exception:
            preserve_failure(
                program,
                dict(
                    kind="malformed_supervisor_verdict",
                    stdout=stdout[-4000:].decode(errors="replace"),
                    stderr=stderr[-4000:].decode(errors="replace"),
                ),
            )
            raise
        return dict(**result, seconds=time.monotonic() - started)


def validate():
    (STATE / "scratch").mkdir(parents=True, exist_ok=True)
    # These handcrafted probes run before any dataset/reference/model programs.
    probes = {
        "correct": "assert 2 + 2 == 4",
        "wrong": "assert False",
        "timeout": "while True: pass",
        "isolation": """import os, socket
assert not os.path.exists('/home')
assert not os.path.exists('/data')
assert not os.path.exists('/root')
assert not os.path.exists('/etc/ssh')
assert not any(k in os.environ for k in ('SSH_AUTH_SOCK','HF_TOKEN','WANDB_API_KEY'))
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try:
        socket.socket(family, socket.SOCK_STREAM)
    except PermissionError: pass
    else: raise AssertionError('socket creation not blocked by kernel')
try:
    open('/usr/sandbox-must-not-write', 'w')
except OSError: pass
else: raise AssertionError('read-only root not enforced')
""",
        "memory_limit": "x = bytearray(2 * 1024**3)",
        "file_creation": "with open('/tmp/test-created-file', 'w') as f: f.write('safe')",
    }
    results = {name: run(code) for name, code in probes.items()}
    assert results["correct"]["passed"] and results["isolation"]["passed"]
    assert results["file_creation"]["passed"]
    assert not results["wrong"]["passed"] and not results["memory_limit"]["passed"]
    assert results["timeout"]["result"] == "timed out"
    return dict(
        passed=True,
        probes=results,
        timeout=3.0,
        memory_bytes=1024**3,
        maximum_file_bytes=1024**2,
        temporary_storage="private tmpfs",
        no_home=True,
        no_network=True,
        network_namespace=False,
        network_isolation="libseccomp kernel syscall filter; socket/io_uring denied",
        filter_sha256=json.loads((STATE / "network-deny.json").read_text())["bpf_sha256"],
        official_execution="HF execute.unsafe_execute; unchanged",
        python=sys.version,
    )


def setup():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Code scoring requires Linux x86_64 with bubblewrap and libseccomp")
    if shutil.which("bwrap") is None:
        raise RuntimeError("Install bubblewrap before running code scoring")
    from .build_seccomp import build

    STATE.mkdir(parents=True, exist_ok=True)
    if not (STATE / "network-deny.json").exists():
        build(STATE)
    return validate()
