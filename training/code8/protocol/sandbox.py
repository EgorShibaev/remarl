"""Minimal read-only runtime, isolated PID/network/filesystem, bounded execution."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
PYTHON_BASE = Path(sys.base_prefix)

def preserve_failure(program, details):
    """Keep reproducible failing input in this Horse root, never home."""
    digest = hashlib.sha256(program.encode()).hexdigest()
    folder = BASE / 'execution-failures' / digest
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'program.py').write_text(program)
    (folder / f'{time.time_ns()}.json').write_text(json.dumps(details, indent=2))

def command(work, entry='/entry.py'):
    return ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
            '--ro-bind', '/usr', '/usr', '--ro-bind', '/bin', '/bin',
            '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
            '--ro-bind', str(PYTHON_BASE), '/opt/python',
            '--ro-bind', str(Path(sys.prefix)/'lib/python3.12/site-packages'), '/opt/extra',
            '--ro-bind', str(BASE / 'official'), '/runner',
            '--ro-bind', str(BASE / 'sandbox_runner.py'), '/entry.py',
            '--ro-bind', str(work), '/work', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp', '--clearenv',
            '--setenv', 'PATH', '/opt/python/bin:/usr/bin:/bin',
            '--setenv', 'TMPDIR', '/tmp', '--setenv', 'OPENBLAS_NUM_THREADS', '1',
            '--chdir', '/work', '/opt/python/bin/python3.12', '-I', entry]

def run(program):
    # Scratch contains ONLY this program, never credentials, checkpoints or other outputs.
    with tempfile.TemporaryDirectory(prefix='code-', dir=BASE / 'scratch') as folder:
        work = Path(folder)
        (work / 'program.py').write_text(program)
        started = time.monotonic()
        process = subprocess.Popen(command(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={'PATH': '/usr/bin:/bin'}, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            # The PID namespace's init is killed: its descendants cannot survive.
            preserve_failure(program, dict(kind='supervisor_outer_timeout', seconds=time.monotonic()-started))
            return dict(passed=False, result='outer timeout', seconds=time.monotonic()-started)
        if process.returncode:
            preserve_failure(program, dict(kind='supervisor_exit', returncode=process.returncode,
                stderr=stderr[-4000:].decode(errors='replace'), seconds=time.monotonic()-started))
            raise RuntimeError(f'sandbox infrastructure exit={process.returncode}: {stderr[-1000:]!r}')
        try:
            result = json.loads(stdout)
            assert type(result['passed']) is bool and isinstance(result['result'], str)
            assert result['passed'] == (result['result'] == 'passed')
            assert result['result'] in ('passed','timed out') or result['result'].startswith('failed:')
        except Exception:
            preserve_failure(program, dict(kind='malformed_supervisor_verdict',
                stdout=stdout[-4000:].decode(errors='replace'), stderr=stderr[-4000:].decode(errors='replace')))
            raise
        return dict(**result, seconds=time.monotonic()-started)

def validate():
    (BASE / 'scratch').mkdir(exist_ok=True)
    # These handcrafted probes run before any dataset/reference/model programs.
    probes = {
        'correct': 'assert 2 + 2 == 4',
        'wrong': 'assert False',
        'timeout': 'while True: pass',
        'isolation': '''import os, socket
assert not os.path.exists('/home')
assert not os.path.exists('/data')
assert not os.path.exists('/root')
assert not os.path.exists('/etc/ssh')
assert not any(k in os.environ for k in ('SSH_AUTH_SOCK','HF_TOKEN','WANDB_API_KEY'))
s = socket.socket()
assert s.connect_ex(('1.1.1.1', 80)) != 0
try:
    open('/usr/sandbox-must-not-write', 'w')
except OSError: pass
else: raise AssertionError('read-only root not enforced')
''',
        'memory_limit': 'x = bytearray(2 * 1024**3)',
        'file_creation': "with open('/tmp/test-created-file', 'w') as f: f.write('safe')",
    }
    results = {name: run(code) for name, code in probes.items()}
    assert results['correct']['passed'] and results['isolation']['passed']
    assert results['file_creation']['passed']
    assert not results['wrong']['passed'] and not results['memory_limit']['passed']
    assert results['timeout']['result'] == 'timed out'
    return dict(passed=True, probes=results, timeout=3.0, memory_bytes=1024**3,
                maximum_file_bytes=1024**2, temporary_storage='private tmpfs',
                no_home=True, no_network=True,
                official_execution='HF execute.unsafe_execute; unchanged', python=sys.version)
