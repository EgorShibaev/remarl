"""Bounded infrastructure retries around the unchanged isolated official executor.

No model-generated code is executed outside the original bwrap/seccomp command.
The inner official program deadline remains3s. This module only gives the outer
process supervisor startup/reaping headroom and limits concurrent supervisors.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time


class InfrastructureStall(RuntimeError):
    pass


class ReliableSandbox:
    def __init__(self, sandbox, evidence, attempts=3, concurrency=2, outer_timeout=30):
        assert attempts==3 and concurrency==2 and outer_timeout==30
        self.sandbox=sandbox
        self.evidence=Path(evidence);self.evidence.mkdir(parents=True,exist_ok=True)
        self.attempts=attempts;self.outer_timeout=outer_timeout
        self.slots=threading.BoundedSemaphore(concurrency)
        self.log_lock=threading.Lock()

    def record(self, data):
        with self.log_lock:
            with (self.evidence/'attempts.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(),**data),allow_nan=False)+'\n')
                f.flush()

    def once(self, program):
        sandbox=self.sandbox
        # Reuse the exact original filter, command, PID namespace and supervisor.
        # Each retry gets a fresh scratch directory and isolated process.
        with tempfile.TemporaryDirectory(prefix='code-retry-',dir=sandbox.BASE/'scratch') as folder:
            work=Path(folder);(work/'program.py').write_text(program)
            started=time.monotonic()
            with sandbox.filter_path().open('rb') as filt:
                p=subprocess.Popen(sandbox.command(work,filt.fileno()),stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(filt.fileno(),),
                    close_fds=True,env={'PATH':'/usr/bin:/bin'},start_new_session=True)
            try:
                stdout,stderr=p.communicate(timeout=self.outer_timeout)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                # Never overlap retries while the old supervisor may still live.
                try:p.communicate(timeout=10)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError('Sandbox cleanup failed; no retry permitted') from exc
                raise InfrastructureStall('outer watchdog expired without official verdict')
            if p.returncode:
                raise InfrastructureStall(f'sandbox infrastructure exit={p.returncode}; stderr={stderr[-1000:]!r}')
            try:
                result=json.loads(stdout)
                valid=(type(result.get('passed')) is bool and isinstance(result.get('result'),str)
                    and result['passed']==(result['result']=='passed')
                    and (result['result'] in ('passed','timed out') or result['result'].startswith('failed:')))
                if not valid:raise ValueError('Invalid official verdict')
            except (ValueError,TypeError,AttributeError) as exc:
                raise InfrastructureStall('missing or malformed supervisor verdict') from exc
            return dict(**result,seconds=time.monotonic()-started)

    def __call__(self, program):
        digest=hashlib.sha256(program.encode()).hexdigest()
        with self.slots:
            for attempt in range(1,self.attempts+1):
                try:
                    result=self.once(program)
                except InfrastructureStall as exc:
                    # Preserve the same input and distinguish infrastructure from
                    # a valid failed/timed-out program. No synthetic zero reward.
                    (self.evidence/(digest+'.py')).write_text(program)
                    self.record(dict(program_sha256=digest,attempt=attempt,kind='infrastructure',error=str(exc)))
                    if attempt==self.attempts:raise
                    time.sleep(.5*attempt)
                else:
                    self.record(dict(program_sha256=digest,attempt=attempt,kind='official_verdict',
                                     passed=result['passed'],result=result['result']))
                    return result
        raise AssertionError('Unreachable')


def install(evidence):
    # code_reward establishes its pinned protocol import path lazily.
    import code_reward
    import sandbox
    repaired=ReliableSandbox(sandbox,evidence)
    sandbox.run=repaired
    code_reward.run=repaired
    assert sandbox.run is code_reward.run
    return repaired
